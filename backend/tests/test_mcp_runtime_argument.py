"""MCP business arguments must survive ToolRuntime injection unchanged."""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import _get_all_injected_args
from mcp.types import CallToolResult, TextContent

from deerflow.mcp import tools as mcp_tools
from deerflow.mcp_scope import mcp_session_scope_key
from deerflow.tools.sync import make_sync_tool_wrapper


@pytest.mark.parametrize("background", [False, True], ids=["pooled", "background"])
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("case", ["required", "optional-present", "optional-omitted", "internal-name-collision"])
def test_mcp_runtime_business_argument(monkeypatch, tmp_path, background, sync, case):
    properties = {"runtime": {"type": "string"}, "code": {"type": "string"}}
    required = ["code"] if case.startswith("optional") else ["runtime", "code"]
    arguments = {"code": "print(1)"}
    if case != "optional-omitted":
        arguments["runtime"] = "python"
    if case == "internal-name-collision":
        for name in ("_deerflow_tool_runtime", "_deerflow_tool_runtime_"):
            properties[name] = {"type": "string"}
            arguments[name] = f"business value for {name}"
            required.append(name)
    schema = {"type": "object", "properties": properties, "required": required}
    original = StructuredTool(name="srv_execute", description="execute code", args_schema=schema, coroutine=AsyncMock())
    captured = {}

    if background:

        async def submit(**kwargs):
            captured["request"] = kwargs["request"]
            return {"id": "task-1", "status": "submitted"}

        monkeypatch.setattr(mcp_tools, "get_mcp_task_submitter", lambda: MagicMock(submit=submit))
        tool = mcp_tools._make_background_submit_tool(original, server_name="srv", task_name="execute", submit_tool="execute", status_tool="status", cancel_tool="cancel", connection_scope="deployment")
    else:
        pool = MagicMock()
        pool.get_session = AsyncMock()

        async def call_tool(*args, **kwargs):
            captured.update(kwargs)
            return CallToolResult(content=[TextContent(type="text", text="ok")])

        monkeypatch.setattr(mcp_tools, "call_pooled_session_tool", call_tool)
        monkeypatch.setattr(mcp_tools, "_prepare_stdio_workspace", lambda *args, **kwargs: (tmp_path, tmp_path, None))
        tool = mcp_tools._make_session_pool_tool(original, "srv", {"transport": "stdio", "command": "unused"}, pool=pool)

    tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)
    graph = StateGraph(MessagesState, context_schema=dict)
    graph.add_node("tools", ToolNode([tool], handle_tool_errors=False))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    compiled = graph.compile()
    message = AIMessage(content="", tool_calls=[{"name": tool.name, "args": arguments, "id": "call-1", "type": "tool_call"}])
    context = {"thread_id": "thread-1", "user_id": "alice", "thread_incarnation": "incarnation-1", "run_id": "run-1"}
    config = {"configurable": {"thread_id": "thread-1"}}
    if sync:
        result = compiled.invoke({"messages": [message]}, config=config, context=context)
    else:
        result = asyncio.run(compiled.ainvoke({"messages": [message]}, config=config, context=context))

    assert result["messages"][-1].status == "success"
    assert tool.args_schema == schema
    if background:
        request = captured["request"]
        assert request.arguments == arguments
        assert (request.user_id, request.thread_id, request.thread_incarnation) == ("alice", "thread-1", "incarnation-1")
        assert (request.run_id, request.tool_call_id) == ("run-1", "call-1")
    else:
        assert captured["arguments"] == arguments
        assert captured["scope_key"] == mcp_session_scope_key(user_id="alice", thread_id="thread-1", thread_incarnation="incarnation-1")
    injected_name = _get_all_injected_args(tool).runtime
    assert injected_name not in properties
    assert injected_name in inspect.signature(tool.func).parameters
