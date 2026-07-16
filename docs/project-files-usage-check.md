# ProjectFileService Usage Check

## Overview

This document lists all known usages of `ProjectFileService` in the project and verifies that they follow the intended API, security constraints, and workflow.

## Usages Found

### 1. MCP server (`code_agent_cli/project_files_mcp_server.py`)

- **Pattern**: Six MCP tools each create a fresh `ProjectFileService` via the `service()` factory.
- **Methods called**:
  - `list_files(pattern, max_files)`
  - `read_file(path, start_line, end_line)`
  - `search_text(query, file_pattern, regex, case_sensitive, max_matches)`
  - `prepare_change(path, content, expected_sha256)`
  - `apply_change(path, content, expected_sha256)`
  - `git_diff(path)`
- **Correctness**: All method signatures match the service definition. Paths are passed as strings; no path traversal is possible because the service validates internally.

### 2. Test suite (`tests/test_project_files.py`)

- **Pattern**: Tests instantiate `ProjectFileService` with a temporary root.
- **Methods called**:
  - `search_text(query)`
  - `prepare_change(path, content)`
  - `apply_change(path, content, expected_sha256)`
- **Correctness**: Tests use a dedicated temporary root, avoiding interference with real files. The `expected_sha256` is obtained from `prepare_change` output.

### 3. `DirectProjectFilesClient` adapter (`tests/test_project_files.py`)

- **Pattern**: A wrapper class that adapts `ProjectFileService` to the same call interface as the MCP client.
- **Methods called**: All six methods via dynamic dispatch (`call(tool, arguments)` → `getattr(self.service, tool)(**arguments)`).
- **Correctness**: The adapter assumes the tool name matches the method name exactly. This is true for the current implementation.

## Security & Constraint Compliance

All usages rely on `ProjectFileService`'s built-in security:

- **Relative paths only**: All callers pass relative paths as strings.
- **No direct access to internal methods**: `_resolve_file`, `_validate_parent_chain` are private and never called from outside.
- **SHA-256 verification**: `prepare_change` and `apply_change` are used in the correct order; tests verify atomic writes.
- **Supported suffixes, size limits, skipped/protected paths**: Enforced by the service, not circumvented.

## Gaps or Deviations

- None found. All usages are consistent with the documented API.

## Conclusion

`ProjectFileService` is used in exactly two places (MCP server and test suite) and both follow the intended security model and workflow. No undocumented or unsafe usage has been identified.
