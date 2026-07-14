from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from code_agent_cli.tokens import estimate_text_tokens


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_SIZE_TOKENS = 700
DEFAULT_CHUNK_OVERLAP_TOKENS = 80
MIN_CHUNK_SIZE_TOKENS = 500
MAX_CHUNK_SIZE_TOKENS = 1000
MIN_CHUNK_OVERLAP_TOKENS = 50
MAX_CHUNK_OVERLAP_TOKENS = 100
DEFAULT_MAX_FILES = 80
DEFAULT_EMBED_BATCH_SIZE = 32
MAX_EMBED_BATCH_SIZE = 128
PAGE_CHAR_ESTIMATE = 1800

TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".py",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}
PDF_EXTENSIONS = {".pdf"}
SKIPPED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


class DocumentIndexError(Exception):
    """Raised when document indexing cannot be completed."""


@dataclass(frozen=True)
class SourceDocument:
    source: str
    title: str
    text: str
    kind: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    strategy: str
    source: str
    title: str
    section: str
    text: str
    start_char: int
    end_char: int

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def default_document_index_db() -> Path:
    configured_path = os.getenv("CODE_AGENT_DOCUMENT_INDEX_DB")
    if configured_path:
        return Path(configured_path).expanduser()

    from code_agent_cli.pipeline_service import default_pipeline_dir

    return default_pipeline_dir() / "document_index.db"


def default_document_index_report() -> Path:
    configured_path = os.getenv("CODE_AGENT_DOCUMENT_INDEX_REPORT")
    if configured_path:
        return Path(configured_path).expanduser()

    from code_agent_cli.pipeline_service import default_pipeline_dir

    return default_pipeline_dir() / "document_index_report.json"


class DocumentIndexService:
    def __init__(
        self,
        *,
        db_path: Path | None = None,
        report_path: Path | None = None,
        ollama_url: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.db_path = db_path or default_document_index_db()
        self.report_path = report_path or default_document_index_report()
        self.ollama_url = (ollama_url or os.getenv("CODE_AGENT_OLLAMA_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
        self.embed_model = embed_model or os.getenv("CODE_AGENT_OLLAMA_EMBED_MODEL") or DEFAULT_OLLAMA_EMBED_MODEL

    def index_path(
        self,
        path: str,
        *,
        strategies: list[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
        overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        max_files: int = DEFAULT_MAX_FILES,
        project_docs_only: bool = False,
        selected_paths: list[str] | None = None,
        embedding_batch_size: int | None = None,
    ) -> dict[str, Any]:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise DocumentIndexError(f"Путь не найден: {root}")
        if chunk_size < MIN_CHUNK_SIZE_TOKENS or chunk_size > MAX_CHUNK_SIZE_TOKENS:
            raise DocumentIndexError(
                "chunk_size должен быть в диапазоне 500-1000 токенов."
            )
        if overlap < MIN_CHUNK_OVERLAP_TOKENS or overlap > MAX_CHUNK_OVERLAP_TOKENS:
            raise DocumentIndexError("overlap должен быть в диапазоне 50-100 токенов.")
        if overlap < 0 or overlap >= chunk_size:
            raise DocumentIndexError("overlap должен быть >= 0 и меньше chunk_size.")
        if max_files < 1:
            raise DocumentIndexError("max_files должен быть положительным числом.")
        batch_size = normalize_embedding_batch_size(embedding_batch_size)

        selected_strategies = normalize_strategies(strategies)
        if selected_paths is not None:
            documents, skipped = load_selected_documents(
                root,
                selected_paths=selected_paths,
                max_files=max_files,
            )
        else:
            documents, skipped = (
                load_project_documents(root, max_files=max_files)
                if project_docs_only
                else load_documents(root, max_files=max_files)
            )
        if not documents:
            if project_docs_only:
                raise DocumentIndexError(
                    "Не найдены README и документы в docs/ или project/docs/."
                )
            raise DocumentIndexError("Не найдено документов для индексации.")

        chunks: list[DocumentChunk] = []
        if "fixed" in selected_strategies:
            chunks.extend(chunk_fixed(documents, chunk_size=chunk_size, overlap=overlap))
        if "structural" in selected_strategies:
            chunks.extend(chunk_structural(documents, chunk_size=chunk_size, overlap=overlap))
        if not chunks:
            raise DocumentIndexError("Chunking не создал ни одного чанка.")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        initialized_at = datetime.now(timezone.utc).isoformat()
        embeddings: list[tuple[DocumentChunk, list[float]]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embed_batch([chunk.text for chunk in batch])
            embeddings.extend(zip(batch, vectors, strict=True))

        save_chunks(
            self.db_path,
            chunks_with_embeddings=embeddings,
            root=root,
            model=self.embed_model,
            initialized_at=initialized_at,
        )

        report = build_report(
            root=root,
            documents=documents,
            chunks=chunks,
            skipped=skipped,
            db_path=self.db_path,
            model=self.embed_model,
            initialized_at=initialized_at,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        report["embedding"]["batch_size"] = batch_size
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report | {"report_path": str(self.report_path)}

    def index_project_docs(
        self,
        path: str,
        *,
        strategies: list[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
        overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        max_files: int = DEFAULT_MAX_FILES,
    ) -> dict[str, Any]:
        """Index only project README files and documentation directories."""
        return self.index_path(
            path,
            strategies=strategies,
            chunk_size=chunk_size,
            overlap=overlap,
            max_files=max_files,
            project_docs_only=True,
        )

    def status(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {
                "exists": False,
                "db_path": str(self.db_path),
                "report_path": str(self.report_path),
            }

        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            total_chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            by_strategy = {
                str(row["strategy"]): int(row["count"])
                for row in connection.execute(
                    "SELECT strategy, COUNT(*) AS count FROM chunks GROUP BY strategy ORDER BY strategy"
                )
            }
            source_count = int(connection.execute("SELECT COUNT(DISTINCT source) FROM chunks").fetchone()[0])
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM index_metadata")
            }

        return {
            "exists": True,
            "db_path": str(self.db_path),
            "report_path": str(self.report_path),
            "chunks": total_chunks,
            "sources": source_count,
            "by_strategy": by_strategy,
            "model": metadata.get("model", ""),
            "root": metadata.get("root", ""),
            "created_at": metadata.get("created_at", ""),
        }

    def compare_chunking(self) -> dict[str, Any]:
        if self.report_path.exists():
            try:
                payload = json.loads(self.report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise DocumentIndexError(f"Отчет chunking поврежден: {error}") from error
            if isinstance(payload, dict):
                return payload

        status = self.status()
        if not status.get("exists"):
            raise DocumentIndexError("Индекс еще не создан. Запустите /mcp index-docs PATH.")
        return status

    def embed(self, text: str) -> list[float]:
        clean_text = text.strip()
        if not clean_text:
            raise DocumentIndexError("Нельзя создать embedding для пустого чанка.")

        payload = json.dumps({"model": self.embed_model, "prompt": clean_text}).encode("utf-8")
        request = Request(
            f"{self.ollama_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise DocumentIndexError(f"Ollama embeddings вернул HTTP {error.code}: {body}") from error
        except URLError as error:
            raise DocumentIndexError(
                f"Ollama недоступна на {self.ollama_url}. Запустите: ollama serve"
            ) from error
        except OSError as error:
            raise DocumentIndexError(f"Ollama embeddings недоступен: {error}") from error
        except json.JSONDecodeError as error:
            raise DocumentIndexError(f"Ollama вернула некорректный JSON: {error}") from error

        embedding = response_payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise DocumentIndexError("Ollama response не содержит embedding.")
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError) as error:
            raise DocumentIndexError("Ollama embedding должен быть массивом чисел.") from error

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts]
        if not clean_texts or any(not text for text in clean_texts):
            raise DocumentIndexError("Нельзя создать embedding для пустого чанка.")

        payload = json.dumps(
            {"model": self.embed_model, "input": clean_texts},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.ollama_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code in {404, 405}:
                return [self.embed(text) for text in clean_texts]
            body = error.read().decode("utf-8", errors="replace")
            raise DocumentIndexError(
                f"Ollama batch embeddings вернул HTTP {error.code}: {body}"
            ) from error
        except URLError as error:
            raise DocumentIndexError(
                f"Ollama недоступна на {self.ollama_url}. Запустите: ollama serve"
            ) from error
        except OSError as error:
            raise DocumentIndexError(f"Ollama batch embeddings недоступен: {error}") from error
        except json.JSONDecodeError as error:
            raise DocumentIndexError(f"Ollama вернула некорректный JSON: {error}") from error

        raw_embeddings = response_payload.get("embeddings")
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(clean_texts):
            raise DocumentIndexError(
                "Ollama batch response содержит неверное количество embeddings."
            )
        embeddings: list[list[float]] = []
        try:
            for embedding in raw_embeddings:
                if not isinstance(embedding, list) or not embedding:
                    raise TypeError
                embeddings.append([float(value) for value in embedding])
        except (TypeError, ValueError) as error:
            raise DocumentIndexError(
                "Ollama batch embedding должен быть массивом чисел."
            ) from error
        return embeddings


def normalize_strategies(strategies: list[str] | None) -> list[str]:
    if not strategies:
        return ["fixed", "structural"]
    normalized = []
    for strategy in strategies:
        clean = strategy.strip().lower()
        if clean not in {"fixed", "structural"}:
            raise DocumentIndexError(f"Неизвестная chunking strategy: {strategy}")
        if clean not in normalized:
            normalized.append(clean)
    return normalized


def normalize_embedding_batch_size(value: int | None) -> int:
    if value is None:
        raw_value = os.getenv("CODE_AGENT_EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE))
        try:
            value = int(raw_value)
        except ValueError as error:
            raise DocumentIndexError(
                "CODE_AGENT_EMBED_BATCH_SIZE должен быть целым числом."
            ) from error
    if value < 1 or value > MAX_EMBED_BATCH_SIZE:
        raise DocumentIndexError(
            f"embedding_batch_size должен быть в диапазоне 1-{MAX_EMBED_BATCH_SIZE}."
        )
    return value


def load_documents(root: Path, *, max_files: int) -> tuple[list[SourceDocument], list[dict[str, str]]]:
    paths = [root] if root.is_file() else list(iter_document_paths(root))
    return load_document_paths(paths, base=root.parent if root.is_file() else root, max_files=max_files)


def load_project_documents(
    root: Path,
    *,
    max_files: int,
) -> tuple[list[SourceDocument], list[dict[str, str]]]:
    if not root.is_dir():
        raise DocumentIndexError(
            "Для индексации документации проекта укажите корневую папку проекта."
        )

    paths: list[Path] = []
    for path in sorted(root.iterdir()):
        if (
            path.is_file()
            and path.name.lower().startswith("readme")
            and path.suffix.lower() in TEXT_EXTENSIONS | PDF_EXTENSIONS
        ):
            paths.append(path)

    for docs_dir in (root / "docs", root / "project" / "docs"):
        if docs_dir.is_dir():
            paths.extend(iter_document_paths(docs_dir))

    unique_paths = list(dict.fromkeys(paths))
    return load_document_paths(unique_paths, base=root, max_files=max_files)


def load_selected_documents(
    root: Path,
    *,
    selected_paths: list[str],
    max_files: int,
) -> tuple[list[SourceDocument], list[dict[str, str]]]:
    if not root.is_dir():
        raise DocumentIndexError("Для выборочной индексации укажите корень проекта.")

    paths: list[Path] = []
    skipped: list[dict[str, str]] = []
    seen: set[Path] = set()
    for relative_value in selected_paths:
        relative_path = Path(relative_value)
        source = relative_path.as_posix()
        if relative_path.is_absolute() or ".." in relative_path.parts:
            skipped.append({"source": source, "reason": "путь вне корня проекта"})
            continue
        candidate = root / relative_path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            skipped.append({"source": source, "reason": "файл не найден"})
            continue
        if not resolved.is_relative_to(root) or not resolved.is_file():
            skipped.append({"source": source, "reason": "путь вне корня проекта"})
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)

    documents, load_skipped = load_document_paths(paths, base=root, max_files=max_files)
    return documents, skipped + load_skipped


def load_document_paths(
    paths: list[Path],
    *,
    base: Path,
    max_files: int,
) -> tuple[list[SourceDocument], list[dict[str, str]]]:
    documents: list[SourceDocument] = []
    skipped: list[dict[str, str]] = []
    for path in paths:
        if len(documents) >= max_files:
            skipped.append({"source": str(path), "reason": f"max_files={max_files}"})
            continue
        try:
            document = load_document(path, base=base)
        except DocumentIndexError as error:
            skipped.append({"source": str(path), "reason": str(error)})
            continue
        if document.text.strip():
            documents.append(document)
    return documents, skipped


def iter_document_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIPPED_DIRS for part in path.parts):
            continue
        if has_hidden_path_part(path, root):
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS | PDF_EXTENSIONS:
            continue
        paths.append(path)
    return paths


def has_hidden_path_part(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    return any(part.startswith(".") for part in relative_parts)


def load_document(path: Path, *, base: Path) -> SourceDocument:
    suffix = path.suffix.lower()
    source = relative_source(path, base)
    if suffix in PDF_EXTENSIONS:
        text = extract_pdf_text(path)
        return SourceDocument(source=source, title=path.name, text=text, kind="pdf")

    if suffix not in TEXT_EXTENSIONS:
        raise DocumentIndexError(f"Неподдерживаемый тип файла: {suffix}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise DocumentIndexError(f"Не удалось прочитать файл: {error}") from error

    return SourceDocument(source=source, title=path.name, text=normalize_text(text), kind=suffix.lstrip(".") or "text")


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as error:
        raise DocumentIndexError("PDF пропущен: установите optional dependency pypdf для чтения PDF.") from error

    try:
        reader = PdfReader(str(path))
        parts = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        raise DocumentIndexError(f"Не удалось извлечь текст из PDF: {error}") from error
    return normalize_text("\n\n".join(parts))


def relative_source(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def chunk_fixed(
    documents: list[SourceDocument],
    *,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        text = document.text.strip()
        if not text:
            continue
        for index, (start, end, chunk_text) in enumerate(
            split_by_estimated_tokens(text, max_tokens=chunk_size, overlap_tokens=overlap)
        ):
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=make_chunk_id("fixed", document.source, index),
                        strategy="fixed",
                        source=document.source,
                        title=document.title,
                        section=document.title,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                    )
                )
    return chunks


def chunk_structural(
    documents: list[SourceDocument],
    *,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        sections = document_sections(document)
        parts: list[tuple[str, int, int, str]] = []
        for section_title, section_text, section_start in sections:
            for part_start, part_end, part_text in split_large_section(section_text, chunk_size=chunk_size, overlap=overlap):
                parts.append(
                    (
                        section_title or document.title,
                        section_start + part_start,
                        section_start + part_end,
                        part_text,
                    )
                )
        for index, (section_title, start, end, text) in enumerate(
            merge_structural_parts(parts, min_tokens=MIN_CHUNK_SIZE_TOKENS, max_tokens=chunk_size)
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id("structural", document.source, index),
                    strategy="structural",
                    source=document.source,
                    title=document.title,
                    section=section_title,
                    text=text,
                    start_char=start,
                    end_char=end,
                )
            )
    return chunks


def merge_structural_parts(
    parts: list[tuple[str, int, int, str]],
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[tuple[str, int, int, str]]:
    merged: list[tuple[str, int, int, str]] = []
    buffer_sections: list[str] = []
    buffer_texts: list[str] = []
    buffer_start: int | None = None
    buffer_end = 0
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer_sections, buffer_texts, buffer_start, buffer_end, buffer_tokens
        if not buffer_texts or buffer_start is None:
            return
        section = " / ".join(unique_preserve_order(buffer_sections))
        text = "\n\n".join(buffer_texts).strip()
        if text:
            merged.append((section, buffer_start, buffer_end, text))
        buffer_sections = []
        buffer_texts = []
        buffer_start = None
        buffer_end = 0
        buffer_tokens = 0

    for section, start, end, text in parts:
        part_tokens = estimate_text_tokens(text)
        if buffer_texts and buffer_tokens >= min_tokens and buffer_tokens + part_tokens > max_tokens:
            flush()
        if buffer_texts and buffer_tokens + part_tokens > max_tokens:
            flush()

        if buffer_start is None:
            buffer_start = start
        buffer_sections.append(section)
        buffer_texts.append(text)
        buffer_end = end
        buffer_tokens += part_tokens

        if buffer_tokens >= min_tokens:
            flush()

    flush()
    return merged


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def document_sections(document: SourceDocument) -> list[tuple[str, str, int]]:
    if document.kind in {"md", "markdown", "rst"}:
        return markdown_sections(document)
    if document.kind == "py":
        return python_sections(document)
    return paragraph_sections(document)


def markdown_sections(document: SourceDocument) -> list[tuple[str, str, int]]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(document.text))
    if not matches:
        return paragraph_sections(document)

    sections: list[tuple[str, str, int]] = []
    if matches[0].start() > 0:
        preface = document.text[: matches[0].start()].strip()
        if preface:
            sections.append((document.title, preface, 0))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
        title = match.group(2).strip()
        text = document.text[start:end].strip()
        if text:
            sections.append((title, text, start))
    return sections


def python_sections(document: SourceDocument) -> list[tuple[str, str, int]]:
    try:
        tree = ast.parse(document.text)
    except SyntaxError:
        return paragraph_sections(document)

    lines = document.text.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and hasattr(node, "lineno")
        and hasattr(node, "end_lineno")
    ]
    nodes.sort(key=lambda node: (node.lineno, getattr(node, "col_offset", 0)))
    sections: list[tuple[str, str, int]] = []
    covered_ranges: list[tuple[int, int]] = []
    for node in nodes:
        start_line = max(int(node.lineno) - 1, 0)
        end_line = max(int(node.end_lineno or node.lineno), int(node.lineno))
        start = line_offsets[start_line] if start_line < len(line_offsets) else 0
        end = line_offsets[end_line] if end_line < len(line_offsets) else len(document.text)
        if any(start >= existing_start and end <= existing_end for existing_start, existing_end in covered_ranges):
            continue
        section_type = "class" if isinstance(node, ast.ClassDef) else "function"
        text = document.text[start:end].strip()
        if text:
            sections.append((f"{section_type} {node.name}", text, start))
            covered_ranges.append((start, end))

    if not sections:
        return paragraph_sections(document)
    return sections


def paragraph_sections(document: SourceDocument) -> list[tuple[str, str, int]]:
    sections: list[tuple[str, str, int]] = []
    cursor = 0
    for raw_part in re.split(r"\n\s*\n", document.text):
        part = raw_part.strip()
        if not part:
            cursor += len(raw_part)
            continue
        start = document.text.find(part, cursor)
        if start < 0:
            start = cursor
        sections.append((document.title, part, start))
        cursor = start + len(part)
    if not sections and document.text.strip():
        sections.append((document.title, document.text.strip(), 0))
    return sections


def split_large_section(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    clean = text.strip()
    if not clean:
        return []
    if estimate_text_tokens(clean) <= chunk_size:
        start = text.find(clean)
        start = max(start, 0)
        return [(start, start + len(clean), clean)]

    return split_by_estimated_tokens(
        text,
        max_tokens=chunk_size,
        overlap_tokens=overlap,
        prefer_boundaries=True,
    )


def split_by_estimated_tokens(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    prefer_boundaries: bool = False,
) -> list[tuple[int, int, str]]:
    pieces = token_pieces(text)
    if not pieces:
        return []

    chunks: list[tuple[int, int, str]] = []
    start_index = 0
    while start_index < len(pieces):
        end_index = start_index
        token_count = 0
        while end_index < len(pieces) and token_count + pieces[end_index][2] <= max_tokens:
            token_count += pieces[end_index][2]
            end_index += 1
        if end_index == start_index:
            end_index += 1

        if prefer_boundaries and end_index < len(pieces):
            boundary_index = find_piece_boundary(text, pieces, start_index, end_index)
            if boundary_index > start_index:
                end_index = boundary_index

        start = pieces[start_index][0]
        end = pieces[end_index - 1][1]
        chunk_text = text[start:end].strip()
        if chunk_text:
            trimmed_start = text.find(chunk_text, start, end)
            trimmed_start = start if trimmed_start < 0 else trimmed_start
            chunks.append((trimmed_start, trimmed_start + len(chunk_text), chunk_text))

        if end_index >= len(pieces):
            break
        start_index = overlap_start_index(pieces, end_index, overlap_tokens)
        if start_index <= 0 and chunks:
            start_index = end_index
    return chunks


def token_pieces(text: str) -> list[tuple[int, int, int]]:
    pieces: list[tuple[int, int, int]] = []
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        value = match.group(0)
        if re.fullmatch(r"\w+", value, flags=re.UNICODE):
            weight = max(1, ceil(len(value) / 4))
        else:
            weight = 1
        pieces.append((match.start(), match.end(), weight))
    return pieces


def overlap_start_index(
    pieces: list[tuple[int, int, int]],
    end_index: int,
    overlap_tokens: int,
) -> int:
    index = max(end_index - 1, 0)
    token_count = 0
    while index > 0 and token_count < overlap_tokens:
        token_count += pieces[index][2]
        index -= 1
    return max(index, 0)


def find_piece_boundary(
    text: str,
    pieces: list[tuple[int, int, int]],
    start_index: int,
    end_index: int,
) -> int:
    start = pieces[start_index][0]
    end = pieces[end_index - 1][1]
    boundary_char = find_boundary(text, start, end)
    if boundary_char <= start:
        return end_index
    for index in range(end_index - 1, start_index, -1):
        if pieces[index][1] <= boundary_char:
            return index + 1
    return end_index


def find_boundary(text: str, start: int, end: int) -> int:
    window = text[start:end]
    for separator in ("\n\n", "\n", ". "):
        position = window.rfind(separator)
        if position >= max(120, len(window) // 2):
            return start + position + len(separator)
    return end


def make_chunk_id(strategy: str, source: str, index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{strategy}:{digest}:{index:04d}"


def save_chunks(
    db_path: Path,
    *,
    chunks_with_embeddings: list[tuple[DocumentChunk, list[float]]],
    root: Path,
    model: str,
    initialized_at: str,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute("DROP TABLE IF EXISTS index_metadata")
        connection.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id, strategy, source, title, section, text, start_char, end_char,
                embedding_json, embedding_dim, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.strategy,
                    chunk.source,
                    chunk.title,
                    chunk.section,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    json.dumps(embedding, ensure_ascii=False),
                    len(embedding),
                    json.dumps(chunk.metadata, ensure_ascii=False),
                )
                for chunk, embedding in chunks_with_embeddings
            ],
        )
        connection.executemany(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
            [
                ("root", str(root)),
                ("model", model),
                ("created_at", initialized_at),
                ("chunks", str(len(chunks_with_embeddings))),
            ],
        )
        connection.execute("CREATE INDEX idx_chunks_strategy ON chunks(strategy)")
        connection.execute("CREATE INDEX idx_chunks_source ON chunks(source)")
        connection.commit()


def build_report(
    *,
    root: Path,
    documents: list[SourceDocument],
    chunks: list[DocumentChunk],
    skipped: list[dict[str, str]],
    db_path: Path,
    model: str,
    initialized_at: str,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    total_chars = sum(len(document.text) for document in documents)
    strategies = sorted({chunk.strategy for chunk in chunks})
    comparison = {
        strategy: strategy_stats([chunk for chunk in chunks if chunk.strategy == strategy])
        for strategy in strategies
    }
    return {
        "created_at": initialized_at,
        "root": str(root),
        "db_path": str(db_path),
        "embedding": {
            "provider": "ollama",
            "model": model,
        },
        "documents": {
            "count": len(documents),
            "total_chars": total_chars,
            "estimated_pages": round(total_chars / PAGE_CHAR_ESTIMATE, 1),
            "sources": [
                {
                    "source": document.source,
                    "title": document.title,
                    "kind": document.kind,
                    "chars": len(document.text),
                }
                for document in documents
            ],
            "skipped": skipped,
        },
        "chunks": {
            "total": len(chunks),
            "target_size_tokens": chunk_size,
            "overlap_tokens": overlap,
            "strategies": comparison,
            "metadata_fields": ["source", "title", "section", "chunk_id", "strategy", "start_char", "end_char"],
        },
        "comparison": compare_strategies(comparison),
        "examples": {
            strategy: [chunk_preview(chunk) for chunk in chunks if chunk.strategy == strategy][:3]
            for strategy in strategies
        },
    }


def strategy_stats(chunks: list[DocumentChunk]) -> dict[str, Any]:
    sizes = [len(chunk.text) for chunk in chunks]
    token_sizes = [estimate_text_tokens(chunk.text) for chunk in chunks]
    sections = {chunk.section for chunk in chunks if chunk.section}
    sources = {chunk.source for chunk in chunks}
    if not sizes:
        return {
            "chunks": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "sources": 0,
            "sections": 0,
        }
    return {
        "chunks": len(chunks),
        "avg_chars": round(sum(sizes) / len(sizes), 1),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "avg_tokens": round(sum(token_sizes) / len(token_sizes), 1),
        "min_tokens": min(token_sizes),
        "max_tokens": max(token_sizes),
        "sources": len(sources),
        "sections": len(sections),
    }


def compare_strategies(stats: dict[str, dict[str, Any]]) -> dict[str, str]:
    fixed = stats.get("fixed", {})
    structural = stats.get("structural", {})
    if not fixed or not structural:
        return {
            "summary": "Построена одна стратегия chunking.",
        }
    return {
        "fixed": (
            "Равномерные чанки удобны для стабильного размера embedding-запросов, "
            "но могут разрезать логические разделы."
        ),
        "structural": (
            "Структурные чанки лучше сохраняют заголовки, функции и смысловые границы, "
            "но размеры получаются менее равномерными."
        ),
        "chunk_count_delta": str(int(structural.get("chunks", 0)) - int(fixed.get("chunks", 0))),
        "section_coverage": (
            f"fixed sections={fixed.get('sections', 0)}, "
            f"structural sections={structural.get('sections', 0)}"
        ),
    }


def chunk_preview(chunk: DocumentChunk) -> dict[str, Any]:
    preview = " ".join(chunk.text.split())
    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "section": chunk.section,
        "chars": len(chunk.text),
        "preview": preview[:240],
    }
