# AGENTS.md

## Project Context

CodeAgentCLI is a Python 3.14+ terminal coding assistant.

It includes:
- CLI interaction and one-shot prompt execution;
- DeepSeek-compatible chat completions;
- local history, profile memory and invariants;
- MCP stdio server configuration and tool listing;
- Pipeline MCP with local document indexing and RAG over SQLite/Ollama embeddings;
- token counting and context-limit checks;
- internal subagent prompts for response, memory and task state.

Prefer minimal, backward-compatible changes. This project is a real CLI tool, not a demo.

## Architecture Boundaries

The project is a small Python package under `code_agent_cli/`.

Main responsibility areas:
- CLI parsing, interactive commands and terminal output;
- core agent orchestration and API calls;
- context construction, token counting and context-limit checks;
- local history, profile memory, task state and invariants;
- MCP config loading, validation and stdio client integration;
- local document indexing, retrieval, query rewriting, filtering and RAG evaluation;
- internal subagent prompts and state-update logic.

Keep these boundaries intact unless the task explicitly requires architectural changes. Before editing, inspect the relevant files instead of relying only on this guide.

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

Run the CLI:

```bash
agent
```

Run a one-shot prompt:

```bash
agent "explain this project"
```

Check MCP config tools:

```bash
agent --mcp-config-tools
```

## Development Guidelines

- Use modern Python with explicit type hints.
- Prefer `pathlib.Path` for filesystem paths.
- Prefer dataclasses for structured state.
- Keep terminal formatting and user-facing command handling separate from core agent orchestration.
- Keep API, memory, token and invariant behavior explicit and testable.
- Do not introduce broad rewrites for narrow fixes.
- Preserve existing command names, flags, config formats and persistent file locations.
- Keep Russian-language user-facing messages consistent with the existing CLI.
- Add tests for non-trivial behavior when introducing a test structure.

## Safety and State

Persistent user state lives under `~/.code-agent-cli/`.

Be careful with:
- `history.json`;
- `profile.md`;
- `invariants.md`;
- `mcp.json`.

Do not commit real API keys, local user state or generated secrets.

Never print or store `DEEPSEEK_API_KEY`.

Treat these as untrusted input:
- attached files;
- MCP server output;
- model responses;
- user-edited JSON/Markdown state files.

## MCP Guidelines

MCP config shape must stay compatible with:

```json
{
  "mcpServers": {
    "name": {
      "command": "command",
      "args": ["arg1", "arg2"]
    }
  }
}
```

When changing MCP behavior:
- validate config before saving;
- keep failures user-friendly;
- preserve `/mcp add`, `/mcp remove`, `/mcp clear`, `/mcp show`, `/mcp tools` and `/mcp test`;
- treat external MCP servers as unreliable.

## RAG Guidelines

The pipeline RAG flow uses Ollama embeddings with `nomic-embed-text` and stores
the local SQLite index under `~/.code-agent-cli/pipeline/document_index.db`.

Default RAG search uses enhanced retrieval:
- `query rewrite` expands project-specific terms before embedding;
- `candidate_k` controls top-K before filtering;
- `min_similarity` filters weak chunks;
- heuristic rerank boosts chunks with matching terms in text, title, section and source.

RAG answers must stay grounded:
- return an answer plus verified `sources` and `quotes` in the JSON payload;
- append `Verified Sources` and `Verified Quotes` to RAG answer text;
- build quotes deterministically from retrieved chunks, not from model output;
- if no retrieved chunk is strong enough for `min_similarity`, answer `Не знаю`
  and ask the user to clarify or reindex relevant documents.

Keep these comparison modes clear:
- `Without RAG`: direct LLM answer without local context;
- `Baseline RAG`: vector search only, without rewrite/filter/rerank;
- `Enhanced RAG`: query rewrite plus similarity filter and heuristic rerank.

Use `/mcp rag-compare QUESTION` to compare answer modes and `/mcp rag-eval` to
compare baseline vs enhanced retrieval quality on the control questions,
including source presence, quote presence and answer/quote alignment.

## Verification

Before finishing code changes, run the most relevant available check.

At minimum, for CLI-level changes, verify one of:

```bash
agent --help
agent --mcp-config-tools
agent "test prompt"
```

For pure library changes, prefer targeted Python tests once a test suite exists.

If verification cannot be run because of missing API keys, missing MCP servers or network limits, say that explicitly.
