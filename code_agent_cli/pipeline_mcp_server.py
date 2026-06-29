from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from code_agent_cli.pipeline_service import PipelineService, default_pipeline_dir


mcp = FastMCP(
    "CodeAgent Pipeline",
    instructions=(
        "MCP server composing tools into a search -> summarize -> save pipeline. "
        "Search uses the web, summarize uses the configured LLM, save writes to disk."
    ),
)


def service() -> PipelineService:
    return PipelineService()


@mcp.tool(
    title="Pipeline health",
    description="Return pipeline output path and service status.",
)
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "output_dir": str(default_pipeline_dir()),
    }


@mcp.tool(
    title="Search web",
    description="Search the internet and return structured web results.",
)
def search(query: str, limit: int = 5) -> dict[str, Any]:
    return service().search(query, limit=limit)


@mcp.tool(
    title="Summarize search results",
    description="Summarize a search payload with the configured LLM.",
)
def summarize(search_payload: dict[str, Any], max_items: int = 5) -> dict[str, Any]:
    return service().summarize(search_payload, max_items=max_items)


@mcp.tool(
    title="Summarize MCP text",
    description="Summarize arbitrary text returned by another MCP tool with the configured LLM.",
)
def summarize_text(query: str, content: str) -> dict[str, Any]:
    return service().summarize_text(query, content)


@mcp.tool(
    title="Save content",
    description="Save content to the pipeline output directory.",
)
def save(filename: str, content: str) -> dict[str, Any]:
    return service().save(filename, content)


@mcp.tool(
    title="Run pipeline",
    description="Automatically run search -> summarize -> save.",
)
def run(query: str, filename: str, limit: int = 5) -> dict[str, Any]:
    return service().run(query, filename, limit=limit)


@mcp.tool(
    title="Index local documents",
    description=(
        "Build a local SQLite document index with Ollama embeddings, metadata, "
        "and fixed plus structural chunking comparison."
    ),
)
def index_documents(
    path: str,
    chunk_size: int = 700,
    overlap: int = 80,
    max_files: int = 80,
) -> dict[str, Any]:
    return service().index_documents(
        path,
        chunk_size=chunk_size,
        overlap=overlap,
        max_files=max_files,
    )


@mcp.tool(
    title="Document index status",
    description="Return status and aggregate counts for the local document index.",
)
def index_status() -> dict[str, Any]:
    return service().index_status()


@mcp.tool(
    title="Compare chunking strategies",
    description="Return the saved comparison between fixed-size and structural chunking.",
)
def compare_chunking() -> dict[str, Any]:
    return service().compare_chunking()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
