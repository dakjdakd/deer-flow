"""Shared IM uploads must reach reused remote sandboxes before publication."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.channels import manager
from app.channels.message_bus import INBOUND_FILE_CONTENT_KEY, InboundMessage, MessageBus
from app.channels.telegram import TelegramChannel
from deerflow.config.paths import Paths
from deerflow.sandbox.lease import discard_sandbox_lease_manager, get_sandbox_lease_manager
from deerflow.uploads.manager import get_uploads_dir


class RemoteProvider:
    """An already-existing VM; acquire never restores host files into it."""

    uses_thread_data_mounts = False

    def __init__(self):
        self.files = {"/mnt/user-data/uploads/old.txt": b"edited in the live sandbox"}
        self.acquired = []
        self.released = []
        self.writes = []
        self.failure = None
        self.started = threading.Event()
        self.finish = threading.Event()
        self.finish.set()
        self.sandbox = SimpleNamespace(id="existing-vm", update_file=self.update_file, release_command_scope=lambda scope: None)

    def acquire(self, thread_id=None, *, user_id=None):
        self.acquired.append((thread_id, user_id))
        if self.failure == "acquire":
            raise RuntimeError("https://secret.invalid/token")
        return "existing-vm"

    async def acquire_async(self, thread_id=None, *, user_id=None):
        return await asyncio.to_thread(self.acquire, thread_id, user_id=user_id)

    def get(self, sandbox_id):
        return None if self.failure == "missing" else self.sandbox

    def get_scoped(self, sandbox_id, *, thread_id, user_id):
        return self.get(sandbox_id)

    def release(self, sandbox_id):
        self.released.append(sandbox_id)

    def update_file(self, path, content):
        self.started.set()
        assert self.finish.wait(10), "test must release the transport worker"
        if self.failure == "update" or path.endswith("bad.txt"):
            raise RuntimeError("https://secret.invalid/token")
        self.writes.append((path, content))
        self.files[path] = content


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr("deerflow.uploads.manager.get_paths", lambda: Paths(tmp_path))
    return tmp_path


@pytest.fixture
def remote(storage):
    provider = RemoteProvider()
    # create=True lets these regressions run against the unfixed manager too.
    with patch.object(manager, "get_sandbox_provider", return_value=provider, create=True):
        yield provider
    provider.finish.set()
    discard_sandbox_lease_manager(provider)


def attachment(*names):
    return InboundMessage(
        channel_name="telegram",
        chat_id="chat",
        user_id="platform-user",
        text="Please read this",
        files=[{"filename": name, INBOUND_FILE_CONTENT_KEY: b"new attachment"} for name in names],
    )


@pytest.mark.asyncio
async def test_reused_remote_receives_final_deduplicated_path_without_replaying_old_files(remote):
    uploads = await asyncio.to_thread(get_uploads_dir, "thread", user_id="owner")
    await asyncio.to_thread(uploads.mkdir, parents=True)
    await asyncio.to_thread((uploads / "old.txt").write_bytes, b"stale host content")
    await asyncio.to_thread((uploads / "report.txt").write_bytes, b"previous report")
    msg = attachment("report.txt")

    created = await manager._ingest_inbound_files("thread", msg, user_id="owner")

    assert len(created) == 1
    saved = created[0]
    assert saved["filename"] == "report_1.txt"
    assert remote.files[saved["path"]] == b"new attachment"
    assert remote.files["/mnt/user-data/uploads/old.txt"] == b"edited in the live sandbox"
    assert remote.writes == [(saved["path"], b"new attachment")]
    assert remote.acquired == [("thread", "owner")]
    assert remote.released == ["existing-vm"]
    assert await asyncio.to_thread((uploads / saved["filename"]).read_bytes) == b"new attachment"
    assert INBOUND_FILE_CONTENT_KEY not in msg.files[0]


@pytest.mark.asyncio
async def test_telegram_downloaded_bytes_reach_reused_remote_sandbox(remote, monkeypatch):
    channel = TelegramChannel(MessageBus(), {"bot_token": "test-token"})
    monkeypatch.setattr(channel, "_get_download_bot", AsyncMock(return_value=object()))
    monkeypatch.setattr(channel, "_download_inbound_file", AsyncMock(return_value=(14, b"new attachment")))

    async def run_download(coro):
        return await coro

    monkeypatch.setattr(channel, "_run_on_telegram_loop", run_download)
    msg = InboundMessage(channel_name="telegram", chat_id="chat", user_id="sender", text="Read this", files=[{"filename": "report.txt", "file_id": "file"}])
    materialized = await channel.receive_file(msg, "thread", user_id="owner")
    created = await manager._ingest_inbound_files("thread", materialized, user_id="owner")
    assert remote.files[created[0]["path"]] == b"new attachment"
    assert INBOUND_FILE_CONTENT_KEY not in materialized.files[0]


@pytest.mark.asyncio
async def test_effective_owner_is_shared_by_host_storage_and_remote_acquisition(remote, monkeypatch):
    monkeypatch.setattr(manager, "get_effective_user_id", lambda: "ambient-owner")
    created = await manager._ingest_inbound_files("thread", attachment("report.txt"))
    uploads = get_uploads_dir("thread", user_id="ambient-owner")
    assert await asyncio.to_thread((uploads / created[0]["filename"]).read_bytes) == b"new attachment"
    assert remote.acquired == [("thread", "ambient-owner")]


@pytest.mark.asyncio
async def test_thread_data_mount_provider_does_not_acquire_or_copy(remote):
    remote.uses_thread_data_mounts = True
    created = await manager._ingest_inbound_files("thread", attachment("report.txt"), user_id="owner")
    assert len(created) == 1
    assert remote.acquired == remote.writes == remote.released == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["acquire", "missing", "update"])
async def test_failed_sync_retains_host_file_without_advertising_missing_path(remote, failure, caplog):
    remote.failure = failure
    msg = attachment("report.txt")
    created = await manager._ingest_inbound_files("thread", msg, user_id="owner")
    assert created == []
    uploads = get_uploads_dir("thread", user_id="owner")
    assert await asyncio.to_thread((uploads / "report.txt").read_bytes) == b"new attachment"
    assert "sandbox" in msg.text.lower() and "failed" in msg.text.lower()
    assert "secret.invalid" not in msg.text + caplog.text


@pytest.mark.asyncio
async def test_failed_attachment_does_not_prevent_later_attachment_sync(remote):
    msg = attachment("bad.txt", "good.txt")
    created = await manager._ingest_inbound_files("thread", msg, user_id="owner")
    assert [entry["filename"] for entry in created] == ["good.txt"]
    assert remote.files[created[0]["path"]] == b"new attachment"
    assert "failed" in msg.text.lower()


@pytest.mark.asyncio
async def test_upload_does_not_park_client_held_by_concurrent_run(remote):
    leases = get_sandbox_lease_manager(remote)
    await leases.acquire_async("run", "thread", user_id="owner")
    try:
        created = await manager._ingest_inbound_files("thread", attachment("report.txt"), user_id="owner")
        assert remote.files[created[0]["path"]] == b"new attachment"
        assert leases.binding_for("run") == "existing-vm"
        assert remote.released == []
    finally:
        await leases.release_async("run")
    assert remote.released == ["existing-vm"]


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_transport_before_releasing_client(remote):
    remote.finish.clear()
    msg = attachment("report.txt")
    task = asyncio.create_task(manager._ingest_inbound_files("thread", msg, user_id="owner"))
    try:
        assert await asyncio.to_thread(remote.started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert remote.released == []
    finally:
        remote.finish.set()
        result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert remote.writes == [("/mnt/user-data/uploads/report.txt", b"new attachment")]
    assert remote.released == ["existing-vm"]
