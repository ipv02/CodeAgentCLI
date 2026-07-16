# ProjectFileService Usage

## Overview

`ProjectFileService` (defined in `code_agent_cli/project_files.py`) is the core
service for safe, atomic file operations within a project root. Its methods
validate paths, prevent directory traversal, disallow symlinks, and enforce
size and type limits.

## Direct Usage

### In the MCP server

The MCP server in `code_agent_cli/project_files_mcp_server.py` creates a fresh
`ProjectFileService` instance per tool call through the `service()` factory:

```python
from code_agent_cli.project_files import ProjectFileService


def service() -> ProjectFileService:
    return ProjectFileService()
```

The instance backs six MCP tools:

- `list_files` calls `service().list_files(...)`;
- `read_file` calls `service().read_file(...)`;
- `search_text` calls `service().search_text(...)`;
- `prepare_change` calls `service().prepare_change(...)`;
- `apply_change` calls `service().apply_change(...)`;
- `git_diff` calls `service().git_diff(...)`.

### In the test suite

Tests in `tests/test_project_files.py` instantiate the service with a temporary
project root:

```python
service = ProjectFileService(root)
service.search_text("BillingAPI")
service.prepare_change("README.md", ...)
service.apply_change("README.md", ..., expected_sha256=...)
```

`DirectProjectFilesClient` adapts the service to the same call interface used
by the MCP client:

```python
class DirectProjectFilesClient:
    def __init__(self, service: ProjectFileService) -> None:
        self.service = service

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        method = getattr(self.service, tool)
        return method(**arguments)
```

This makes goal-level assistant tests deterministic and independent of a
separate stdio process.

## Security and constraints

- Paths must be relative and cannot contain `..` components.
- Resolved paths must remain inside the configured project root.
- Symlinks in the path chain are rejected.
- Protected files such as `.env`, `history.json`, `profile.md`,
  `invariants.md`, and `mcp.json` are inaccessible.
- Only allowlisted text file extensions are accepted.
- Input files and replacement content are limited to 256 KB.
- SHA-256 digests prevent unnoticed concurrent modifications.
- Replacement content is written to a temporary file and installed through
  atomic `os.replace`.

## Typical workflow

1. `search_text(query)` locates relevant files.
2. `read_file(path)` loads the selected source context and its SHA-256 digest.
3. `prepare_change(path, content)` validates content and returns a unified diff
   without writing.
4. `apply_change(path, content, expected_sha256)` writes only if the source SHA
   still matches.
5. `git_diff(path)` returns the resulting read-only Git diff when needed.

## Example

```python
prepared = service.prepare_change("README.md", "# Demo\n\nUpdated\n")
assert "Updated" in prepared["diff"]

applied = service.apply_change(
    "README.md",
    prepared["content"],
    expected_sha256=prepared["expected_sha256"],
)
assert applied["applied"]
```
