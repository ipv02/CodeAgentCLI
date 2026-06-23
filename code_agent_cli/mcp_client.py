from __future__ import annotations

import os
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPConnectionError(Exception):
    """Raised when an MCP server cannot be reached or returns invalid data."""


@dataclass(frozen=True)
class MCPTool:
    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPToolCallResult:
    content: list[dict[str, Any]]
    structured_content: dict[str, Any] | None = None
    is_error: bool = False

    def as_text(self) -> str:
        if self.structured_content is not None:
            return json.dumps(
                self.structured_content,
                ensure_ascii=False,
                indent=2,
            )

        text_values = [
            item["text"]
            for item in self.content
            if item.get("type") == "text" and isinstance(item.get("text"), str)
        ]
        if text_values:
            return "\n".join(text_values)

        return json.dumps(self.content, ensure_ascii=False, indent=2)


async def list_mcp_tools(
    command: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> list[MCPTool]:
    """Connect to an MCP server over stdio and return its advertised tools."""

    try:
        process_env = {**os.environ, **env} if env else None
        server_params = StdioServerParameters(
            command=command,
            args=args,
            cwd=str(cwd) if cwd else None,
            env=process_env,
        )

        async with AsyncExitStack() as exit_stack:
            errlog = exit_stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            transport = await exit_stack.enter_async_context(
                stdio_client(server_params, errlog=errlog)
            )
            read_stream, write_stream = transport
            session = await exit_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=timeout),
                )
            )
            await session.initialize()
            result = await session.list_tools()
            return [_parse_tool(tool) for tool in result.tools]
    except Exception as error:
        raise MCPConnectionError(format_connection_error(error)) from error


async def call_mcp_tool(
    command: str,
    args: list[str],
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> MCPToolCallResult:
    """Connect to an MCP server over stdio and call one advertised tool."""

    try:
        process_env = {**os.environ, **env} if env else None
        server_params = StdioServerParameters(
            command=command,
            args=args,
            cwd=str(cwd) if cwd else None,
            env=process_env,
        )

        async with AsyncExitStack() as exit_stack:
            errlog = exit_stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            transport = await exit_stack.enter_async_context(
                stdio_client(server_params, errlog=errlog)
            )
            read_stream, write_stream = transport
            session = await exit_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=timeout),
                )
            )
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})
            return _parse_tool_call_result(result)
    except Exception as error:
        raise MCPConnectionError(format_connection_error(error)) from error


def format_connection_error(error: BaseException) -> str:
    if isinstance(error, ExceptionGroup):
        messages = [format_connection_error(child) for child in error.exceptions]
        return "; ".join(message for message in messages if message) or str(error)

    message = str(error).strip()
    if message:
        return message

    return error.__class__.__name__


def _parse_tool(raw_tool: Any) -> MCPTool:
    name = getattr(raw_tool, "name", None)
    if not isinstance(name, str) or not name:
        raise MCPConnectionError("MCP-сервер вернул tool без корректного name.")

    title = getattr(raw_tool, "title", None)
    description = getattr(raw_tool, "description", None)
    input_schema = getattr(raw_tool, "inputSchema", None) or {"type": "object"}

    if title is not None and not isinstance(title, str):
        raise MCPConnectionError(f"Tool {name} содержит некорректный title.")
    if description is not None and not isinstance(description, str):
        raise MCPConnectionError(f"Tool {name} содержит некорректный description.")
    if not isinstance(input_schema, dict):
        raise MCPConnectionError(f"Tool {name} содержит некорректный inputSchema.")

    return MCPTool(
        name=name,
        title=title,
        description=description,
        input_schema=input_schema,
    )


def _parse_tool_call_result(raw_result: Any) -> MCPToolCallResult:
    content = getattr(raw_result, "content", None)
    structured_content = getattr(raw_result, "structuredContent", None)
    is_error = bool(getattr(raw_result, "isError", False))

    if not isinstance(content, list):
        raise MCPConnectionError("MCP-сервер вернул некорректный call_tool content.")

    parsed_content = [_parse_content_item(item) for item in content]
    if structured_content is not None and not isinstance(structured_content, dict):
        raise MCPConnectionError("MCP-сервер вернул некорректный structuredContent.")

    return MCPToolCallResult(
        content=parsed_content,
        structured_content=structured_content,
        is_error=is_error,
    )


def _parse_content_item(raw_item: Any) -> dict[str, Any]:
    if hasattr(raw_item, "model_dump"):
        payload = raw_item.model_dump(mode="json", exclude_none=True)
        if isinstance(payload, dict):
            return payload

    item_type = getattr(raw_item, "type", None)
    if isinstance(item_type, str):
        payload: dict[str, Any] = {"type": item_type}
        text = getattr(raw_item, "text", None)
        if isinstance(text, str):
            payload["text"] = text
        return payload

    raise MCPConnectionError("MCP-сервер вернул некорректный content item.")
