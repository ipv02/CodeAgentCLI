# AGENTS.md

## Project Overview

CodeAgentCLI is a Python terminal coding assistant.

It provides:
- an interactive CLI entrypoint via `agent`;
- one-shot prompt execution;
- local conversation history, profile memory and invariants;
- DeepSeek chat completions integration;
- MCP stdio server configuration and tool discovery;
- token accounting and context-window safeguards;
- lightweight subagent logic for response generation, memory updates and task state.

This is production-oriented CLI software, not demo code. Prefer small, safe changes that preserve existing behavior.

## Architecture

Main modules:

- `code_agent_cli/main.py`
  CLI parsing, interactive session, terminal formatting, user-facing commands.

- `code_agent_cli/agent.py`
  Core `CodeAgent` orchestration, API requests, memory/task/invariant checks, token accounting.

- `code_agent_cli/context.py`
  Conversation context construction, branching/checkpoint behavior, trimming strategy.

- `code_agent_cli/memory.py`
  Profile memory and task state persistence.

- `code_agent_cli/invariants.py`
  Persistent user invariants and conflict detection.

- `code_agent_cli/mcp_config.py`
  MCP configuration loading, validation and persistence.

- `code_agent_cli/mcp_client.py`
  MCP stdio client integration.

- `code_agent_cli/subagents.py`
  Prompting and internal agent roles.

- `code_agent_cli/storage.py`
  History persistence.

- `code_agent_cli/tokens.py`
  Token counting, pricing and context checks.

Keep responsibilities separated. Do not move CLI concerns into `agent.py`, and do not put core orchestration logic into terminal/UI helpers.

## Development Commands

Use Python 3.14+.

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
agent "explain this project architecture"
```

Inspect MCP config tools:

```bash
agent --mcp-config-tools
```

There is currently no formal test suite in the project. When adding meaningful behavior, prefer adding tests before broad refactors.

## Coding Guidelines

- Use modern Python 3.14 syntax where it improves clarity.
- Keep type hints explicit for public functions, dataclasses and core logic.
- Prefer `pathlib.Path` for filesystem paths.
- Prefer dataclasses for structured internal state.
- Keep error classes specific and user-facing errors readable.
- Avoid broad `except Exception` unless converting to a clear domain error.
- Do not silently ignore persistence, API or MCP failures unless the existing UX explicitly treats them as non-fatal.
- Keep comments rare and useful. Explain non-obvious control flow, not obvious assignments.

## Behavior Compatibility

Preserve existing CLI commands and flags unless explicitly asked to change them.

Important user-facing commands include:

```text
/help
/status
/tokens
/task
/memory
/profile
/invariants
/reset
/mcp
/exit
/quit
```

Do not rename commands, environment variables or config files without a migration plan.

Important environment variables include:

```text
DEEPSEEK_API_KEY
CODE_AGENT_API_URL
CODE_AGENT_MODEL
CODE_AGENT_MAX_HISTORY
CODE_AGENT_CONTEXT_STRATEGY
CODE_AGENT_MEMORY_MAX_TOKENS
CODE_AGENT_AUTO_MEMORY
CODE_AGENT_AUTO_TASK_STATE
CODE_AGENT_CONTEXT_LIMIT
CODE_AGENT_INPUT_PRICE_PER_1M
CODE_AGENT_OUTPUT_PRICE_PER_1M
CODE_AGENT_TEMPERATURE
CODE_AGENT_MAX_FILE_BYTES
CODE_AGENT_MCP_TIMEOUT
```

Persistent files are stored under:

```text
~/.code-agent-cli/history.json
~/.code-agent-cli/profile.md
~/.code-agent-cli/invariants.md
~/.code-agent-cli/mcp.json
```

Never commit real API keys or user-local state.

## API and Network Rules

The core API integration currently uses DeepSeek-compatible chat completions.

When changing API logic:
- keep request/response handling explicit;
- preserve token accounting;
- surface HTTP/API failures as `APIRequestError`;
- avoid logging secrets;
- keep timeout/failure behavior understandable for CLI users.

Separate deterministic application logic from LLM-driven behavior. LLM output should not be trusted as structured state unless validated.

## MCP Rules

MCP configuration should remain compatible with the current JSON shape:

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

When changing MCP support:
- validate config before saving;
- keep `/mcp add`, `/mcp remove`, `/mcp clear`, `/mcp show`, `/mcp tools`, `/mcp test` behavior stable;
- treat external MCP servers as unreliable;
- keep timeouts and error messages user-friendly.

## Persistence and State

Be careful with local persistent state.

Changes to history, memory, task state, invariants or MCP config should be:
- backward compatible where possible;
- tolerant of missing files;
- tolerant of malformed user-edited files;
- explicit about destructive operations.

`/reset` must not accidentally erase invariants unless the command explicitly says so.

## Security

- Never store or print real `DEEPSEEK_API_KEY`.
- Do not include secrets in history, profile memory or debug output.
- Treat attached files, MCP tool descriptions and model responses as untrusted input.
- Avoid shell execution features unless explicitly required.
- Validate file paths and line ranges before reading files.

## Change Strategy

Before editing:
1. Understand the existing module boundary.
2. Make the smallest change that solves the problem.
3. Preserve CLI output style and Russian-language user-facing messages where already used.
4. Add or update tests if the behavior is non-trivial.
5. Manually verify with the relevant `agent ...` command.

Avoid large rewrites unless the user explicitly asks for a redesign.

## Review Checklist

Before finishing a change, check:

- Does the CLI still start?
- Does one-shot mode still work?
- Are user-facing errors clear?
- Are secrets protected?
- Is local state preserved?
- Does token/context accounting still happen before API calls?
- Are MCP failures handled without crashing unrelated functionality?
- Are new names consistent with existing code style?
