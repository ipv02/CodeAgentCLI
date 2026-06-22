from __future__ import annotations

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
        server_params = StdioServerParameters(
            command=command,
            args=args,
            cwd=str(cwd) if cwd else None,
            env=env,
        )

        async with AsyncExitStack() as exit_stack:
            transport = await exit_stack.enter_async_context(stdio_client(server_params))
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
        raise MCPConnectionError(str(error)) from error


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
