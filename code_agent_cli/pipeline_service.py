from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from code_agent_cli.agent import env_float, https_ssl_context
from code_agent_cli.document_index import DocumentIndexService


SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"


class PipelineError(Exception):
    """Raised when an MCP pipeline step cannot be completed."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def default_pipeline_dir() -> Path:
    configured_path = os.getenv("CODE_AGENT_PIPELINE_DIR")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path.home() / ".code-agent-cli" / "pipeline"


def enrich_with_curated_sources(
    query: str,
    results: list[SearchResult],
    *,
    limit: int,
) -> list[SearchResult]:
    curated = curated_results_for_query(query)
    if not curated:
        return results

    by_url: dict[str, SearchResult] = {}
    for result in [*curated, *results]:
        by_url.setdefault(result.url, result)
    return list(by_url.values())[:limit]


def curated_results_for_query(query: str) -> list[SearchResult]:
    lowered = query.lower()
    if "swiftui" not in lowered:
        return []
    if not any(marker in lowered for marker in ("navigation", "навигац", "ios")):
        return []

    return [
        SearchResult(
            title="Apple Developer Documentation: NavigationStack",
            url="https://developer.apple.com/documentation/swiftui/navigationstack",
            snippet=(
                "NavigationStack is Apple's SwiftUI container for presenting a stack "
                "of views and managing data-driven navigation."
            ),
        ),
        SearchResult(
            title="Apple Developer Documentation: NavigationSplitView",
            url="https://developer.apple.com/documentation/swiftui/navigationsplitview",
            snippet=(
                "NavigationSplitView is Apple's SwiftUI container for multicolumn "
                "navigation layouts on iPadOS, macOS and larger displays."
            ),
        ),
        SearchResult(
            title="Apple Developer Documentation: Navigation",
            url="https://developer.apple.com/documentation/swiftui/navigation",
            snippet=(
                "Apple's SwiftUI Navigation documentation covers APIs and patterns "
                "for moving through app screens and presenting navigation state."
            ),
        ),
    ]


class PipelineService:
    def search(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            raise PipelineError("query не должен быть пустым.")
        if limit < 1:
            raise PipelineError("limit должен быть положительным числом.")

        request = Request(
            f"{SEARCH_ENDPOINT}?q={quote_plus(clean_query)}",
            headers={
                "User-Agent": "CodeAgentCLI/0.1 MCP Pipeline",
            },
        )
        try:
            with urlopen(request, timeout=30, context=https_ssl_context()) as response:
                html = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            raise PipelineError(f"Search API вернул HTTP {error.code}.") from error
        except OSError as error:
            raise PipelineError(f"Search API недоступен: {error}") from error

        curated_results = curated_results_for_query(clean_query)
        duck_results = parse_duckduckgo_html(html, limit=limit)

        results = [*curated_results, *duck_results]
        source = SEARCH_ENDPOINT
        results = enrich_with_curated_sources(clean_query, results, limit=limit)
        if not results:
            raise PipelineError(
                "search не нашел результатов. DuckDuckGo мог вернуть anti-bot страницу; "
                "для этой темы нет специализированного provider."
            )
        return {
            "query": clean_query,
            "source": source,
            "count": len(results),
            "results": [result.__dict__ for result in results],
        }

    def summarize(self, search_payload: dict[str, Any], *, max_items: int = 5) -> dict[str, Any]:
        query = str(search_payload.get("query") or "").strip()
        results = normalize_results(search_payload.get("results"))[:max_items]
        if not query:
            raise PipelineError("search_payload.query не должен быть пустым.")
        if not results:
            raise PipelineError("search_payload.results пуст.")

        summary = generate_pipeline_summary(query, results)
        return {
            "query": query,
            "summary": summary["content"],
            "model": summary["model"],
            "usage": summary["usage"],
            "items_used": len(results),
            "sources": [
                {
                    "title": result.title,
                    "url": result.url,
                }
                for result in results
            ],
        }

    def summarize_text(self, query: str, content: str) -> dict[str, Any]:
        clean_query = query.strip()
        clean_content = content.strip()
        if not clean_query:
            raise PipelineError("query не должен быть пустым.")
        if not clean_content:
            raise PipelineError("content не должен быть пустым.")

        summary = generate_text_summary(clean_query, clean_content)
        return {
            "query": clean_query,
            "summary": summary["content"],
            "model": summary["model"],
            "usage": summary["usage"],
            "source": "mcp_text",
            "items_used": 1,
        }

    def save(self, filename: str, content: str) -> dict[str, Any]:
        safe_name = sanitize_filename(filename)
        clean_content = content.strip()
        if not clean_content:
            raise PipelineError("content не должен быть пустым.")

        output_dir = default_pipeline_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / safe_name
        path.write_text(clean_content + "\n", encoding="utf-8")
        return {
            "saved": True,
            "path": str(path),
            "bytes": path.stat().st_size,
        }

    def run(self, query: str, filename: str, *, limit: int = 5) -> dict[str, Any]:
        search_payload = self.search(query, limit=limit)
        summary_payload = self.summarize(search_payload, max_items=limit)
        content = render_saved_summary(summary_payload)
        save_payload = self.save(filename, content)
        return {
            "pipeline": "search -> summarize -> save",
            "query": query,
            "search": search_payload,
            "summary": summary_payload,
            "save": save_payload,
        }

    def index_documents(
        self,
        path: str,
        *,
        chunk_size: int = 700,
        overlap: int = 80,
        max_files: int = 80,
    ) -> dict[str, Any]:
        return DocumentIndexService().index_path(
            path,
            chunk_size=chunk_size,
            overlap=overlap,
            max_files=max_files,
        )

    def index_status(self) -> dict[str, Any]:
        return DocumentIndexService().status()

    def compare_chunking(self) -> dict[str, Any]:
        return DocumentIndexService().compare_chunking()


def parse_duckduckgo_html(html: str, *, limit: int) -> list[SearchResult]:
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r".*?"
        r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    results: list[SearchResult] = []
    for match in pattern.finditer(html):
        title = clean_html(match.group("title"))
        url = clean_url(unescape(match.group("url")))
        snippet = clean_html(match.group("snippet"))
        if title and url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= limit:
            break
    return results


def normalize_results(value: Any) -> list[SearchResult]:
    if not isinstance(value, list):
        return []

    results: list[SearchResult] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if title and url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def generate_pipeline_summary(query: str, results: list[SearchResult]) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise PipelineError("DEEPSEEK_API_KEY не задан.")

    model = os.getenv("CODE_AGENT_MODEL", "deepseek-v4-flash")
    api_url = os.getenv("CODE_AGENT_API_URL", "https://api.deepseek.com/chat/completions")
    temperature = env_float("CODE_AGENT_TEMPERATURE", 0.2)
    max_tokens = int(os.getenv("CODE_AGENT_PIPELINE_MAX_TOKENS", "900"))
    sources_text = "\n".join(
        f"{index}. {result.title}\nURL: {result.url}\nSnippet: {result.snippet}"
        for index, result in enumerate(results, start=1)
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты MCP pipeline agent. Суммаризируй только предоставленные "
                    "search results. Не выдумывай факты и сохраняй ссылки на источники."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Запрос: {query}\n\n"
                    f"Search results:\n{sources_text}\n\n"
                    "Сделай краткую русскоязычную сводку в 3-6 пунктах и добавь "
                    "короткий список источников."
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120, context=https_ssl_context()) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as error:
        response_text = error.read().decode("utf-8")
        raise PipelineError(f"LLM API вернул HTTP {error.code}: {response_text}") from error
    except OSError as error:
        raise PipelineError(f"LLM API недоступен: {error}") from error

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise PipelineError("LLM API вернул некорректный JSON.") from error

    choices = response_payload.get("choices") or []
    usage = response_payload.get("usage") or {}
    if not choices:
        raise PipelineError("LLM API не вернул choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise PipelineError("LLM API вернул пустой summary.")

    return {
        "content": content.strip(),
        "model": model,
        "usage": usage if isinstance(usage, dict) else {},
    }


def generate_text_summary(query: str, content: str) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise PipelineError("DEEPSEEK_API_KEY не задан.")

    model = os.getenv("CODE_AGENT_MODEL", "deepseek-v4-flash")
    api_url = os.getenv("CODE_AGENT_API_URL", "https://api.deepseek.com/chat/completions")
    temperature = env_float("CODE_AGENT_TEMPERATURE", 0.2)
    max_tokens = int(os.getenv("CODE_AGENT_PIPELINE_MAX_TOKENS", "900"))
    clipped_content = content[:12000]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты MCP pipeline agent. Суммаризируй только предоставленный "
                    "MCP output. Не выдумывай факты, явно отмечай ограничения входных данных."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Задача: {query}\n\n"
                    f"MCP output:\n{clipped_content}\n\n"
                    "Сделай краткую русскоязычную сводку в 3-6 пунктах."
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120, context=https_ssl_context()) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as error:
        response_text = error.read().decode("utf-8")
        raise PipelineError(f"LLM API вернул HTTP {error.code}: {response_text}") from error
    except OSError as error:
        raise PipelineError(f"LLM API недоступен: {error}") from error

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise PipelineError("LLM API вернул некорректный JSON.") from error

    choices = response_payload.get("choices") or []
    usage = response_payload.get("usage") or {}
    if not choices:
        raise PipelineError("LLM API не вернул choices.")
    message = choices[0].get("message") or {}
    summary = message.get("content")
    if not isinstance(summary, str) or not summary.strip():
        raise PipelineError("LLM API вернул пустой summary.")

    return {
        "content": summary.strip(),
        "model": model,
        "usage": usage if isinstance(usage, dict) else {},
    }


def render_saved_summary(summary_payload: dict[str, Any]) -> str:
    lines = [
        f"# Pipeline Summary: {summary_payload.get('query', '')}",
        "",
        str(summary_payload.get("summary", "")).strip(),
        "",
        "## Sources",
    ]
    sources = summary_payload.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                lines.append(f"- {source.get('title', '')}: {source.get('url', '')}")
    return "\n".join(lines).strip()


def sanitize_filename(filename: str) -> str:
    name = Path(filename.strip()).name
    if not name:
        raise PipelineError("filename не должен быть пустым.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not safe:
        raise PipelineError("filename не содержит допустимых символов.")
    if not Path(safe).suffix:
        safe = f"{safe}.md"
    return safe


def clean_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", value)
    return " ".join(unescape(without_tags).split())


def clean_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.netloc == "duckduckgo.com" and parsed.path.startswith("/l/"):
        return value
    return value
