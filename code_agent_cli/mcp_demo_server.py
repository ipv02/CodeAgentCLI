from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("code-agent-cli-demo-mcp", log_level="WARNING")


@mcp.tool(title="Echo")
def echo(text: str) -> str:
    """Return the provided text unchanged."""

    return text


@mcp.tool(title="Current Time")
def current_time() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    mcp.run(transport="stdio")
