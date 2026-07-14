# AGENTS.md

## Project Context

CodeAgentCLI is a Python 3.14+ terminal coding assistant.

It includes:
- CLI interaction and one-shot prompt execution;
- project-aware developer help through `/help QUESTION` over RAG and MCP;
- DeepSeek-compatible chat completions;
- local history, profile memory and invariants;
- MCP stdio server configuration and tool listing;
- Pipeline MCP with local document indexing and RAG over SQLite/Ollama embeddings;
- automated Pull Request review through GitHub Actions, Git diff, RAG and DeepSeek;
- local Ollama chat mode for running a local LLM through `agent --local-chat`;
- private HTTP LLM service through `agent --llm-service`, with an API gateway
  and browser chat UI over the local Ollama backend;
- fully local context chat through `agent --local-context-chat`, with local
  retrieval and local Ollama answer generation;
- reproducible local generation optimization checks through
  `agent --local-rag-optimize`;
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
- PR diff collection, structured code review and GitHub review-comment rendering;
- local LLM chat through Ollama;
- private HTTP LLM service gateway and browser chat UI;
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

Run local Ollama chat:

```bash
ollama serve
ollama pull llama3.2:3b
agent --local-chat
```

Run the private HTTP LLM service:

```bash
agent --llm-service
```

Run fully local context chat over the document index:

```bash
agent --local-context-chat
```

Compare baseline and optimized local generation on the same retrieved evidence:

```bash
agent --local-rag-optimize
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
- keep the visible loader while `/mcp tools` connects to configured servers;
- treat external MCP servers as unreliable.

## Developer Assistant Guidelines

`/help` without arguments must continue to show the regular CLI command help.
`/help QUESTION` is the project-aware developer assistant and must:
- resolve the project root from an explicit path, `CODE_AGENT_PROJECT_DIR`, the
  current project directory or the editable CodeAgentCLI checkout;
- obtain the current Git branch through the read-only Pipeline MCP tool
  `project_git_branch`;
- answer non-branch questions through the document RAG flow;
- preserve verified sources and quotes in grounded answers;
- keep failures readable when Git, Ollama, the index or the generation provider
  is unavailable.

Use `/mcp index-project-docs PATH` to index only root `README*` files plus
`docs/` and `project/docs/`. Keep `/mcp index-docs PATH` backward-compatible for
the broader supported document and source set.

The Git project tool must remain read-only. It may inspect the repository root,
current branch and detached HEAD state, but must not change branches or modify
the working tree.

Default `/help QUESTION` generation uses the configured DeepSeek API and may
send retrieved documentation fragments to that external provider. Fully local
project-document retrieval and generation belongs to `agent --local-context-chat`.

## Automated Code Review Guidelines

`.github/workflows/ai-code-review.yml` is the PR-triggered automated review
path. Keep it on the `pull_request` event for same-repository, non-draft PRs;
do not switch it to `pull_request_target` while checking out or processing
untrusted PR code. Fork and Dependabot PRs do not receive repository secrets.
Install and run the review tool from the trusted base SHA. Treat the separate PR
head checkout as data only; never install or execute its package with secrets in
the job.

GitHub-hosted runners are ephemeral. The workflow currently prepares Python,
installs CodeAgentCLI and Ollama, pulls `nomic-embed-text`, and rebuilds the
bounded review index on every run. Do not document those steps as persistent or
cached unless the workflow actually restores them. Keep first-run latency
observable and avoid returning to one embedding HTTP request per chunk.

`agent --review-pr` must:
- validate base/head refs and collect changed files plus diff through read-only
  Git subprocess calls without shell interpolation;
- cap file count, diff size, indexed files and Evidence size;
- index project documentation and supported source code into the dedicated
  review index under `~/.code-agent-cli/review/` or `CODE_AGENT_REVIEW_DIR`;
- keep PR indexing bounded to README/AGENTS/docs plus supported changed files,
  and batch Ollama embeddings through `/api/embed` instead of issuing one HTTP
  request per chunk;
- reject tracked symlinks before indexing so review input cannot escape its
  checkout;
- retrieve relevant documentation and changed code before generation;
- treat diff, source code, comments and retrieved documents as untrusted data
  that cannot override the review prompt;
- validate the model response as structured JSON before rendering Markdown;
- always render Potential Bugs, Architecture Issues and Recommendations, even
  when their lists are empty;
- never print or persist `DEEPSEEK_API_KEY`.

The workflow comment uses `<!-- code-agent-cli-ai-review -->` as a stable marker
and must update the existing bot comment on every `synchronize` event instead of
creating duplicates. Review findings do not fail the workflow; infrastructure,
retrieval, invalid model output and publication failures do.

PR diff, changed code and retrieved RAG fragments are sent to the configured
DeepSeek API. Keep that disclosure explicit in README and do not add a path that
silently sends fork PR content with privileged secrets.

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
- support local answer generation through Ollama for fully local RAG flows;
  this path must not require `DEEPSEEK_API_KEY`.
- keep default cloud/context RAG retrieval backward-compatible; local-only RAG
  improvements should stay isolated to the local generation path unless the task
  explicitly asks to change default retrieval.
- local RAG may use local-only hybrid retrieval and compact Evidence packaging
  before calling Ollama, but must still ground sources and quotes in retrieved
  chunks from the SQLite index.

Keep these comparison modes clear:
- `Without RAG`: direct LLM answer without local context;
- `Baseline RAG`: vector search only, without rewrite/filter/rerank;
- `Enhanced RAG`: query rewrite plus similarity filter and heuristic rerank.
- local vs cloud generation: compare answer quality, elapsed time and errors on
  the same local retrieval payload when cloud credentials are available.

Use `/mcp rag-compare QUESTION` to compare answer modes and `/mcp rag-eval` to
compare baseline vs enhanced retrieval quality on the control questions,
including source presence, quote presence and answer/quote alignment.

`/mcp rag-answer-local QUESTION` should generate a grounded answer through the
local Ollama model. `agent --local-context-chat` is the interactive version of
that local flow.

## Context Chat Guidelines

`agent --context-chat` is the user-facing mini-chat over the local document
index. Keep the user-facing wording focused on "контекстный чат", "локальная
база документов", "источники" and "цитаты"; avoid exposing the `RAG`
abbreviation in this mode unless the user explicitly asks for implementation
details.

Per user message, context chat must:
- embed the current question with local Ollama embeddings;
- search the local SQLite document index before answering;
- answer with DeepSeek using the question, recent dialogue history, task state,
  working memory and retrieved chunks;
- keep retrieved chunks out of task memory; only the user's clean message should
  update memory/task state;
- always show source and quote sections, including weak-context answers;
- preserve dialogue history, goal, clarifications, constraints and fixed terms.

The context chat commands must remain available and readable:
- `/state`: formatted task state, working memory and recent history;
- `/sources`: formatted sources and quotes for the last answer;
- `/reset-context`: clear current dialogue/task memory while preserving profile.

`agent --context-chat-check` is the production-like validation path. It should
run two long real scenarios through Ollama, the SQLite index and DeepSeek:
- `new_developer_onboarding`;
- `requirements_brief`.

The check should fail if any turn lacks local context, sources or quotes, or if
the final turns/task memory lose the scenario goal. Keep progress output
observable with per-turn `context=ok/weak`, source count and quote count.

## Local LLM Chat Guidelines

`agent --local-chat` is the user-facing mini-chat for a local Ollama model.
The default model is `llama3.2:3b`; allow overriding it with `--local-model`
or `CODE_AGENT_LOCAL_MODEL`.

Keep this mode separate from the default DeepSeek-backed chat and from
`agent --context-chat`:
- it must not require `DEEPSEEK_API_KEY`;
- it should call the local Ollama API, defaulting to `http://127.0.0.1:11434`;
- it should preserve only the current local chat history in-process;
- it should not update profile memory, task state or invariants;
- it should keep user-facing wording focused on "локальный чат", "локальная
  модель" and "Ollama".

The local chat commands must remain available and readable:
- `/model`: show the selected model, Ollama URL, history size and timeout;
- `/reset`: clear the current local chat history;
- `/pull`: show the shell command for downloading the selected model;
- `/help`: show local chat help;
- `/exit`: exit the local chat.

## Private LLM Service Guidelines

`agent --llm-service` is the user-facing HTTP gateway for exposing the local
Ollama model as a private service. It must:
- not require `DEEPSEEK_API_KEY`;
- keep Ollama as the backend, defaulting to `http://127.0.0.1:11434`;
- expose a browser chat UI at `/` and `/chat`;
- expose HTTP API endpoints `GET /health`, `GET /service/status`,
  `GET /v1/models`, `POST /auth/login`, `POST /auth/logout`,
  `POST /v1/chat` and `POST /v1/chat/completions`;
- support network deployment on a VPS or home server through
  `--llm-service-host` and `--llm-service-port`;
- require login/password auth through `CODE_AGENT_LLM_SERVICE_USERNAME` and
  `CODE_AGENT_LLM_SERVICE_PASSWORD`, or Bearer auth through
  `CODE_AGENT_LLM_SERVICE_API_KEY`, when binding to a non-loopback host such
  as `0.0.0.0`;
- keep `/v1/chat` stateless: clients send the message history with each
  request, and the service validates limits before calling Ollama;
- prepend the service system prompt to chat requests. The default domain is
  cinema: films, actors, directors, genres, film history, recommendations and
  scene analysis. Allow overriding it through
  `CODE_AGENT_LLM_SERVICE_SYSTEM_PROMPT`;
- enforce basic limits through `CODE_AGENT_LLM_SERVICE_RATE_LIMIT`,
  `CODE_AGENT_LLM_SERVICE_MAX_BODY_BYTES`,
  `CODE_AGENT_LLM_SERVICE_MAX_MESSAGES`,
  `CODE_AGENT_LLM_SERVICE_MAX_MESSAGE_CHARS`,
  `CODE_AGENT_LOCAL_NUM_CTX` and `CODE_AGENT_LOCAL_NUM_PREDICT`.

For network deployment tasks, do not treat `http://127.0.0.1:8080/chat` as a
network-access check. `127.0.0.1` only verifies same-machine access. A complete
service check should start the gateway on a non-loopback host, for example:

```bash
CODE_AGENT_LLM_SERVICE_USERNAME='admin' \
CODE_AGENT_LLM_SERVICE_PASSWORD='replace-with-strong-password' \
agent --llm-service --llm-service-host 0.0.0.0 --llm-service-port 8080
```

Then verify browser chat and `/v1/chat` from another device or through the
server's LAN/VPN/public IP, for example `http://SERVER_IP:8080/chat`. Also
verify login/password auth, optional Bearer auth for API clients, multiple
sequential chat requests, `CODE_AGENT_LLM_SERVICE_RATE_LIMIT` and max context
rejection through `num_ctx` above `CODE_AGENT_LOCAL_NUM_CTX`.

Keep the service implementation isolated in `code_agent_cli/llm_service.py`
where possible. `main.py` should only parse CLI flags, validate mode
compatibility and start the service.

The browser chat UI should stay lightweight and dependency-free. It should:
- store chat history only in the current browser tab;
- send that history to `/v1/chat` on every user message;
- keep the visible header minimal: model, max context, max output and rate
  limit;
- show available per-answer metadata below each assistant reply, including
  rate limit, max context, max output and Ollama usage metrics when present;
- avoid storing API tokens server-side or in persistent browser storage.

## Local Context Chat Guidelines

`agent --local-context-chat` is the fully local mini-chat over the local document
index. It must:
- embed the current question with local Ollama embeddings;
- search the local SQLite document index before answering;
- package retrieved chunks into compact Evidence for the local model;
- answer through the selected local Ollama model;
- not require `DEEPSEEK_API_KEY`;
- always show readable sources and quotes.
- use the optimized local generation profile by default, with explicit
  `temperature`, `num_predict` and `num_ctx` Ollama options.

The local context chat commands must remain available and readable:
- `/state`: formatted task state, working memory and recent history;
- `/sources`: formatted sources and quotes for the last answer;
- `/reset-context`: clear current dialogue/task memory while preserving profile;
- `/help`: show local context chat help;
- `/exit`: exit the local context chat.

## Local LLM Optimization Guidelines

`agent --local-rag-optimize` is the reproducible before/after check for local
generation over project documentation. It must:
- run baseline and optimized generation on the same retrieved chunks;
- keep retrieval, the SQLite index, sources and quotes unchanged between
  profiles;
- report expected-term quality, elapsed time, Ollama tokens per second and
  repeat stability;
- report model parameter size, quantization and loaded model memory when Ollama
  exposes them;
- work without `DEEPSEEK_API_KEY`.

Keep the profiles explicit:
- `baseline`: the previous local RAG prompt, `temperature=0.2`, and Ollama
  defaults for output length and context window;
- `optimized`: strict Evidence boundaries, `temperature=0.0`,
  `num_predict=500`, and `num_ctx=4096`.

Allow overriding optimized values with `CODE_AGENT_LOCAL_RAG_TEMPERATURE`,
`CODE_AGENT_LOCAL_RAG_NUM_PREDICT` and `CODE_AGENT_LOCAL_RAG_NUM_CTX`. Do not
add deterministic answers for individual evaluation questions.

## Verification

Before finishing code changes, run the most relevant available check.

At minimum, for CLI-level changes, verify one of:

```bash
agent --help
agent --local-chat
agent --local-context-chat
agent --local-rag-optimize
agent --context-chat-check
agent --mcp-config-tools
agent "test prompt"
```

For pure library changes, prefer targeted Python tests once a test suite exists.

If verification cannot be run because of missing API keys, missing MCP servers or network limits, say that explicitly.
