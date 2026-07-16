from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any


MAX_PROJECT_FILES = 300
MAX_FILE_BYTES = 256_000
MAX_SEARCH_MATCHES = 250
MAX_CHANGE_BYTES = 256_000
SUPPORTED_SUFFIXES = {
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".py",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".swift",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
}
SKIPPED_PARTS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
PROTECTED_NAMES = {
    ".env",
    "history.json",
    "profile.md",
    "invariants.md",
    "mcp.json",
}


class ProjectFileError(RuntimeError):
    """Raised when a project file operation is invalid or unsafe."""


def default_project_files_root() -> Path:
    configured = os.getenv("CODE_AGENT_PROJECT_FILES_ROOT")
    root = Path(configured).expanduser() if configured else Path.cwd()
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ProjectFileError(f"Корень проекта не найден: {resolved}")
    return resolved


@dataclass(frozen=True)
class ProjectFileChange:
    path: str
    content: str
    expected_sha256: str
    diff: str = ""


class ProjectFileService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_project_files_root()).expanduser().resolve()
        if not self.root.is_dir():
            raise ProjectFileError(f"Корень проекта не найден: {self.root}")

    def list_files(self, *, pattern: str = "", max_files: int = 120) -> dict[str, Any]:
        limit = normalize_limit(max_files, maximum=MAX_PROJECT_FILES, field="max_files")
        clean_pattern = pattern.strip().lower()
        paths: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if len(paths) >= limit:
                break
            if not self._is_supported_file(path):
                continue
            relative = path.relative_to(self.root).as_posix()
            if clean_pattern and clean_pattern not in relative.lower():
                continue
            paths.append(relative)
        return {"root": str(self.root), "count": len(paths), "files": paths}

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_file(path, must_exist=True)
        if start_line < 1:
            raise ProjectFileError("start_line должен быть положительным числом.")
        if end_line is not None and end_line < start_line:
            raise ProjectFileError("end_line должен быть не меньше start_line.")
        text = self._read_text(resolved)
        lines = text.splitlines()
        selected_end = len(lines) if end_line is None else min(end_line, len(lines))
        selected = "\n".join(lines[start_line - 1 : selected_end])
        return {
            "path": resolved.relative_to(self.root).as_posix(),
            "start_line": start_line,
            "end_line": selected_end,
            "total_lines": len(lines),
            "sha256": sha256_text(text),
            "content": selected,
        }

    def search_text(
        self,
        query: str,
        *,
        file_pattern: str = "",
        regex: bool = False,
        case_sensitive: bool = False,
        max_matches: int = 100,
    ) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query or len(clean_query) > 500:
            raise ProjectFileError("query должен содержать от 1 до 500 символов.")
        limit = normalize_limit(
            max_matches,
            maximum=MAX_SEARCH_MATCHES,
            field="max_matches",
        )
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            expression = re.compile(clean_query if regex else re.escape(clean_query), flags)
        except re.error as error:
            raise ProjectFileError(f"Некорректное регулярное выражение: {error}") from error

        matches: list[dict[str, Any]] = []
        scanned_files = 0
        for relative in self.list_files(pattern=file_pattern, max_files=MAX_PROJECT_FILES)["files"]:
            path = self._resolve_file(str(relative), must_exist=True)
            scanned_files += 1
            for line_number, line in enumerate(self._read_text(path).splitlines(), start=1):
                if expression.search(line):
                    matches.append(
                        {
                            "path": str(relative),
                            "line": line_number,
                            "text": line.strip()[:500],
                        }
                    )
                    if len(matches) >= limit:
                        return search_payload(clean_query, scanned_files, matches, truncated=True)
        return search_payload(clean_query, scanned_files, matches, truncated=False)

    def prepare_change(
        self,
        path: str,
        content: str,
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_file(path, must_exist=False)
        clean_content = normalize_content(content)
        encoded = clean_content.encode("utf-8")
        if len(encoded) > MAX_CHANGE_BYTES:
            raise ProjectFileError(f"Новое содержимое превышает {MAX_CHANGE_BYTES} байт.")
        exists = resolved.exists()
        old_content = self._read_text(resolved) if exists else ""
        old_sha = sha256_text(old_content) if exists else ""
        if expected_sha256 is not None and expected_sha256 != old_sha:
            raise ProjectFileError("Файл изменился после чтения; подготовка изменения отменена.")
        relative = resolved.relative_to(self.root).as_posix()
        diff = render_file_diff(relative, old_content, clean_content, exists=exists)
        return {
            "path": relative,
            "exists": exists,
            "changed": old_content != clean_content,
            "expected_sha256": old_sha,
            "new_sha256": sha256_text(clean_content),
            "content": clean_content,
            "diff": diff,
        }

    def apply_change(
        self,
        path: str,
        content: str,
        *,
        expected_sha256: str,
    ) -> dict[str, Any]:
        prepared = self.prepare_change(
            path,
            content,
            expected_sha256=expected_sha256,
        )
        if not prepared["changed"]:
            return {**prepared, "applied": False, "reason": "no changes"}
        resolved = self._resolve_file(path, must_exist=False)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._validate_parent_chain(resolved)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            dir=resolved.parent,
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(str(prepared["content"]))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, resolved)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {**prepared, "applied": True}

    def git_diff(self, *, path: str = "") -> dict[str, Any]:
        arguments = ["git", "diff", "--no-ext-diff", "--"]
        if path.strip():
            resolved = self._resolve_file(path, must_exist=False)
            arguments.append(resolved.relative_to(self.root).as_posix())
        try:
            result = subprocess.run(
                arguments,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            raise ProjectFileError(f"Не удалось получить Git diff: {error}") from error
        return {"root": str(self.root), "diff": result.stdout[:100_000]}

    def _resolve_file(self, value: str, *, must_exist: bool) -> Path:
        clean = value.strip()
        relative = Path(clean)
        if not clean or relative.is_absolute() or ".." in relative.parts:
            raise ProjectFileError("Путь должен быть относительным и находиться внутри проекта.")
        candidate = self.root / relative
        try:
            resolved = candidate.resolve(strict=must_exist)
        except OSError as error:
            raise ProjectFileError(f"Файл не найден: {relative.as_posix()}") from error
        if not resolved.is_relative_to(self.root):
            raise ProjectFileError("Путь выходит за пределы проекта.")
        if any(part in SKIPPED_PARTS for part in relative.parts):
            raise ProjectFileError("Доступ к служебному каталогу запрещён.")
        if relative.name.lower() in PROTECTED_NAMES:
            raise ProjectFileError("Доступ к защищённому файлу запрещён.")
        if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ProjectFileError(f"Неподдерживаемый тип файла: {resolved.suffix or '<без расширения>'}")
        self._validate_parent_chain(candidate)
        if must_exist and not resolved.is_file():
            raise ProjectFileError(f"Файл не найден: {relative.as_posix()}")
        return resolved

    def _validate_parent_chain(self, path: Path) -> None:
        current = path
        while current != self.root:
            if current.exists() and current.is_symlink():
                raise ProjectFileError("Доступ через symlink запрещён.")
            current = current.parent

    def _is_supported_file(self, path: Path) -> bool:
        if not path.is_file() or path.is_symlink():
            return False
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return False
        if any(part in SKIPPED_PARTS for part in relative.parts):
            return False
        if relative.name.lower() in PROTECTED_NAMES:
            return False
        return path.suffix.lower() in SUPPORTED_SUFFIXES and path.stat().st_size <= MAX_FILE_BYTES

    def _read_text(self, path: Path) -> str:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ProjectFileError(f"Файл превышает лимит {MAX_FILE_BYTES} байт.")
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ProjectFileError(f"Не удалось прочитать файл: {error}") from error


def normalize_limit(value: int, *, maximum: int, field: str) -> int:
    if value < 1 or value > maximum:
        raise ProjectFileError(f"{field} должен быть в диапазоне 1-{maximum}.")
    return value


def normalize_content(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if not normalized or normalized.endswith("\n") else normalized + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_file_diff(path: str, old: str, new: str, *, exists: bool) -> str:
    return "".join(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}" if exists else "/dev/null",
            tofile=f"b/{path}",
        )
    )


def search_payload(
    query: str,
    scanned_files: int,
    matches: list[dict[str, Any]],
    *,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "query": query,
        "scanned_files": scanned_files,
        "match_count": len(matches),
        "truncated": truncated,
        "matches": matches,
    }
