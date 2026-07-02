from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import shlex
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import readline  # noqa: F401
except ImportError:
    readline = None

from code_agent_cli.agent import (
    APIRequestError,
    CodeAgentError,
    CodeAgent,
    ContextLimitExceededError,
    MissingAPIKeyError,
)
from code_agent_cli.mcp_config import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    add_mcp_server,
    clear_mcp_servers,
    default_apple_mcp_config_payload,
    default_mcp_config_file,
    load_mcp_config,
    load_mcp_config_or_empty,
    remove_mcp_server,
    save_default_apple_mcp_config,
)
from code_agent_cli.mcp_client import (
    MCPConnectionError,
    MCPTool,
    MCPToolCallResult,
    call_mcp_tool,
    list_mcp_tools,
)
from code_agent_cli.subagents import (
    MCPOrchestrationPlan,
    MCPOrchestrationStep,
    MCPToolDescriptor,
)
from code_agent_cli.tokens import TokenBreakdown


RESET = "\033[0m"
BOLD = "\033[1m"
MUTED = "\033[38;5;245m"
SUBTLE = "\033[38;5;239m"
ACCENT = "\033[38;5;81m"
ACCENT_SOFT = "\033[38;5;110m"
USER_INPUT = "\033[38;5;214m"
SUCCESS = "\033[38;5;114m"
ERROR = "\033[38;5;203m"
WARNING = "\033[38;5;215m"
VALUE = "\033[38;5;159m"
MONEY = "\033[38;5;120m"
COMMAND = "\033[38;5;81m"
ANSWER_TEXT = "\033[38;5;252m"
ANSWER_MUTED = MUTED
ANSWER_ACCENT = ACCENT_SOFT
CODE_BORDER = "\033[38;5;60m"
CODE_TEXT = "\033[38;5;253m"
CODE_STRING = "\033[38;5;180m"
CODE_NUMBER = "\033[38;5;149m"
CODE_KEYWORD = "\033[38;5;141m"
DEFAULT_MAX_FILE_BYTES = 120 * 1024
DEFAULT_WRAP_WIDTH = 96
MIN_WRAP_WIDTH = 56

CODE_KEYWORDS = {
    "and",
    "as",
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "def",
    "do",
    "else",
    "enum",
    "except",
    "false",
    "finally",
    "for",
    "from",
    "func",
    "function",
    "guard",
    "if",
    "import",
    "in",
    "let",
    "nil",
    "none",
    "null",
    "private",
    "public",
    "return",
    "self",
    "static",
    "struct",
    "switch",
    "throw",
    "throws",
    "true",
    "try",
    "var",
    "while",
}


@dataclass(frozen=True)
class FileRange:
    start: int
    end: int


@dataclass(frozen=True)
class PromptPayload:
    request_text: str
    history_text: str | None = None


@dataclass(frozen=True)
class MCPServerCheck:
    server: MCPServerConfig
    tools: list[MCPTool]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class MCPOrchestrationToolCatalog:
    tools: list[MCPToolDescriptor]
    checks: list[MCPServerCheck]


@dataclass(frozen=True)
class MCPOrchestrationStepResult:
    index: int
    step: MCPOrchestrationStep
    arguments: dict[str, Any]
    result: MCPToolCallResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None and not self.result.is_error


@dataclass(frozen=True)
class MCPOrchestrationRunResult:
    plan: MCPOrchestrationPlan
    steps: list[MCPOrchestrationStepResult]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if args.mcp_tools is not None:
        if not run_mcp_tools(args.mcp_tools, args.mcp_timeout):
            raise SystemExit(1)
        return

    if args.mcp_config_tools:
        if not run_mcp_config_tools(args.mcp_config, args.mcp_timeout):
            raise SystemExit(1)
        return

    if args.mcp_init_apple:
        if not init_apple_mcp_config(args.mcp_config, args.mcp_force):
            raise SystemExit(1)
        return

    agent = CodeAgent()

    if args.prompt:
        prompt = build_prompt(args)
        if not send(agent, prompt):
            raise SystemExit(1)
        return

    run_interactive_session(agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Standalone coding assistant for the terminal.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="One-shot prompt. If omitted, starts an interactive chat.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Attach a source file to the prompt.",
    )
    parser.add_argument(
        "--range",
        dest="line_range",
        help="Attach only a line range from --file, for example 40:120.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=env_int("CODE_AGENT_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES),
        help="Warn before sending files larger than this size. Defaults to 120 KB.",
    )
    parser.add_argument(
        "--force-file",
        action="store_true",
        help="Send a large attached file without asking for confirmation.",
    )
    parser.add_argument(
        "--mcp-tools",
        type=Path,
        metavar="SERVER",
        help="Connect to an MCP stdio server script and print its tools.",
    )
    parser.add_argument(
        "--mcp-timeout",
        type=float,
        default=env_float("CODE_AGENT_MCP_TIMEOUT", 30.0),
        help="MCP response timeout in seconds. Defaults to 30.",
    )
    parser.add_argument(
        "--mcp-config",
        type=Path,
        default=default_mcp_config_file(),
        help="Path to MCP config. Defaults to ~/.code-agent-cli/mcp.json.",
    )
    parser.add_argument(
        "--mcp-config-tools",
        action="store_true",
        help="Connect to all MCP servers from config and print their tools.",
    )
    parser.add_argument(
        "--mcp-init-apple",
        action="store_true",
        help="Create MCP config with apple-mcp and cupertino servers.",
    )
    parser.add_argument(
        "--mcp-force",
        action="store_true",
        help="Overwrite existing MCP config when used with --mcp-init-apple.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.line_range and args.file is None:
        parser.error("--range можно использовать только вместе с --file.")

    if args.force_file and args.file is None:
        parser.error("--force-file можно использовать только вместе с --file.")

    if args.max_file_bytes < 1:
        parser.error("--max-file-bytes должен быть положительным числом.")

    if args.mcp_timeout <= 0:
        parser.error("--mcp-timeout должен быть положительным числом.")

    mcp_mode_enabled = (
        args.mcp_tools is not None
        or args.mcp_config_tools
        or args.mcp_init_apple
    )
    if not mcp_mode_enabled:
        return

    modes = [
        args.mcp_tools is not None,
        args.mcp_config_tools,
        args.mcp_init_apple,
    ]
    if sum(1 for enabled in modes if enabled) > 1:
        parser.error("MCP-режимы нельзя использовать одновременно.")

    if args.prompt:
        parser.error("MCP-режим нельзя совмещать с prompt.")

    if args.file is not None or args.line_range or args.force_file:
        parser.error("MCP-режим нельзя совмещать с --file, --range или --force-file.")

    if args.mcp_force and not args.mcp_init_apple:
        parser.error("--mcp-force можно использовать только вместе с --mcp-init-apple.")

    if args.mcp_tools is not None:
        server_path = args.mcp_tools
        if not server_path.exists():
            parser.error(f"MCP server script не найден: {server_path}")
        if server_path.suffix.lower() not in {".py", ".js"}:
            parser.error("MCP server script должен быть .py или .js.")


def run_mcp_tools(server_path: Path, timeout: float) -> bool:
    command, args = build_mcp_server_command(server_path)
    return run_mcp_tools_command(command, args, timeout)


def run_mcp_config_tools(config_path: Path, timeout: float) -> bool:
    try:
        config = load_mcp_config(config_path)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return False

    return print_mcp_config_tools(config, timeout)


def init_apple_mcp_config(config_path: Path, force: bool) -> bool:
    try:
        saved_path = save_default_apple_mcp_config(config_path, overwrite=force)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return False
    except OSError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return False

    print(header_line("MCP"))
    print(status_line("Конфиг создан", str(saved_path), SUCCESS))
    print()
    print(indented_line("Серверы:"))
    print(indented_line("- apple-mcp: bunx --no-cache apple-mcp@latest", level=2))
    print(indented_line("- cupertino: cupertino serve --no-reap", level=2))
    return True


def run_mcp_tools_command(command: str, args: list[str], timeout: float) -> bool:
    try:
        tools = asyncio.run(list_mcp_tools(command, args, timeout=timeout))
    except FileNotFoundError as error:
        print(f"Ошибка: команда MCP-сервера не найдена: {error.filename}", file=sys.stderr)
        return False
    except MCPConnectionError as error:
        print(f"Ошибка MCP: {error}", file=sys.stderr)
        return False
    except Exception as error:
        print(f"Ошибка MCP: {error}", file=sys.stderr)
        return False

    print_mcp_tools(tools)
    return True


def print_mcp_config_tools(config: MCPConfig, timeout: float) -> bool:
    print(header_line("MCP"))
    print(status_line("Конфиг", str(config.path), VALUE))
    print(status_line("Серверов", str(len(config.servers)), VALUE))

    if not config.servers:
        return True

    checks = check_mcp_config_servers(config, timeout)
    print()

    for index, check in enumerate(checks, start=1):
        print_mcp_server_tools(check)
        if index < len(config.servers):
            print()

    success_count = sum(1 for check in checks if check.ok)
    total_tools = sum(len(check.tools) for check in checks)
    print()
    print(status_line("Connected servers", f"{success_count} / {len(config.servers)}", SUCCESS if success_count == len(config.servers) else ERROR))
    print(status_line("Инструментов", str(total_tools), VALUE))
    return success_count == len(config.servers)


def print_mcp_config_missing(config_path: Path, error: MCPConfigError | None = None) -> None:
    print(header_line("MCP"))
    print(status_line("Конфиг", str(config_path), VALUE))
    if error is None:
        print(status_line("Статус", "серверы еще не настроены", WARNING))
    else:
        print(status_line("Статус", str(error), WARNING))
    print()
    print(indented_line("Добавить MCP-сервер:"))
    print(indented_line("/mcp add NAME -- COMMAND ARG1 ARG2", level=2))
    print()
    print(indented_line("Примеры:"))
    print(indented_line("/mcp add apple-mcp -- bunx --no-cache apple-mcp@latest", level=2))
    print(indented_line("/mcp add cupertino -- cupertino serve --no-reap", level=2))
    print()
    print(indented_line("После добавления проверьте подключение:"))
    print(indented_line("/mcp", level=2))


def print_mcp_config_servers(config: MCPConfig) -> None:
    print(header_line("MCP"))
    print(status_line("Конфиг", str(config.path), VALUE))
    print(status_line("Серверов", str(len(config.servers)), VALUE))
    if not config.servers:
        print()
        print(indented_line("Серверы еще не настроены."))
        print(indented_line("/mcp add NAME -- COMMAND ARG1 ARG2", level=2))
        return

    print()
    for server in config.servers:
        print(command_line(server.name))
        print(indented_line(shlex.join([server.command, *server.args])))
        if server.cwd is not None:
            print(indented_line(f"cwd: {server.cwd}"))
        if server.env:
            print(indented_line(f"env: {len(server.env)} переменных"))


def print_mcp_help() -> None:
    print_command_help_grouped_section(
        "MCP",
        (
            (
                "Config",
                (
                    ("/mcp", "проверить подключение серверов из config"),
                    ("/mcp tools", "показать инструменты серверов"),
                    ("/mcp show", "показать сохраненные серверы"),
                    ("/mcp add NAME -- COMMAND ARGS", "добавить или подключить свой MCP-сервер"),
                    ("/mcp remove NAME", "удалить MCP-сервер из config"),
                    ("/mcp clear", "удалить все MCP-серверы из config"),
                    ("/mcp path", "показать путь к config"),
                    ("/mcp test", "проверить подключение с диагностикой ошибок"),
                    ("/mcp call SERVER TOOL JSON", "вызвать MCP-инструмент напрямую"),
                    ("/mcp help", "показать помощь по MCP"),
                    ("agent --mcp-config-tools", "проверить MCP config из shell"),
                ),
            ),
            (
                "Init",
                (
                    ("/mcp init-apple", "создать config для apple-mcp и cupertino"),
                    ("/mcp init-mock", "подключить встроенный mock HTTP API MCP-сервер"),
                    ("/mcp init-scheduler", "подключить встроенный SQLite MCP-планировщик"),
                    ("/mcp init-pipeline", "подключить web+LLM MCP pipeline"),
                    ("/mcp init-orchestration", "подключить apple-mcp, cupertino, pipeline и scheduler"),
                ),
            ),
            (
                "Scheduler",
                (
                    ("/mcp remind TEXT AT", "создать reminder без JSON"),
                    ("/mcp run_due", "выполнить due jobs"),
                    ("/mcp summary", "показать сводку scheduler"),
                    ("/mcp clear-scheduler", "очистить jobs и историю scheduler"),
                ),
            ),
            (
                "Pipeline и RAG",
                (
                    ("/mcp pipeline QUERY FILE", "запустить search -> summarize -> save"),
                    ("/mcp index-docs PATH", "построить локальный индекс документов через Ollama embeddings"),
                    ("/mcp index-status", "показать статус локального индекса документов"),
                    ("/mcp compare-chunking", "сравнить fixed и structural chunking"),
                    ("/mcp rag-search QUESTION", "enhanced search: query rewrite, similarity filter и heuristic rerank"),
                    ("/mcp rag-answer QUESTION", "ответить с verified sources/quotes или сказать Не знаю"),
                    ("/mcp rag-compare QUESTION", "сравнить Without RAG, Baseline RAG и Enhanced RAG"),
                    ("/mcp rag-eval", "проверить sources, quotes и answer/quote alignment на 10 вопросах"),
                    ("/mcp orchestrate TEXT", "построить и выполнить multi-server MCP flow"),
                ),
            ),
        ),
    )
    print()
    print(subheader_line("Примеры"))
    print_help_examples(
        (
            "/mcp init-mock",
            "/mcp init-scheduler",
            "/mcp init-pipeline",
            "/mcp init-orchestration",
            '/mcp remind "Проверить планировщик" 2026-06-24T12:30:00Z',
            "/mcp run_due",
            "/mcp summary",
            "/mcp clear-scheduler",
            '/mcp pipeline "latest MCP protocol news" mcp-summary.md',
            "/mcp index-docs .",
            "/mcp compare-chunking",
            '/mcp rag-search "Какая Ollama модель используется для embeddings?"',
            '/mcp rag-compare "Где хранится MCP config?"',
            "/mcp rag-eval",
            '/mcp orchestrate "найди лучшие практики навигации SwiftUI в iOS через Apple/Cupertino MCP, сохрани в заметки и поставь напоминание проверить завтра"',
            '/mcp call mock-api get_mock_user {"user_id": 1}',
            '/mcp call scheduler summary {"limit": 5}',
            "/mcp add apple-mcp -- bunx --no-cache apple-mcp@latest",
            "/mcp add cupertino -- cupertino serve --no-reap",
        )
    )
    print()
    print_ollama_help()


def add_mcp_server_from_command(config_path: Path, argument: str) -> None:
    try:
        parts = shlex.split(argument)
    except ValueError as error:
        print(f"Ошибка MCP: {error}", file=sys.stderr)
        return

    if len(parts) < 4 or parts[0] != "add" or parts[2] != "--":
        print("Использование: /mcp add NAME -- COMMAND ARG1 ARG2")
        return

    name = parts[1]
    command = parts[3]
    args = parts[4:]
    server = MCPServerConfig(name=name, command=command, args=args)

    try:
        saved_path = add_mcp_server(config_path, server)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        print("Для замены удалите старый сервер: /mcp remove NAME")
        return
    except OSError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return

    print(header_line("MCP"))
    print(status_line("Сервер добавлен", name, SUCCESS))
    print(status_line("Конфиг", str(saved_path), VALUE))
    print()
    print(indented_line("Проверить подключение:"))
    print(indented_line("/mcp", level=2))


def remove_mcp_server_from_command(config_path: Path, argument: str) -> None:
    try:
        parts = shlex.split(argument)
    except ValueError as error:
        print(f"Ошибка MCP: {error}", file=sys.stderr)
        return

    if len(parts) != 2 or parts[0] != "remove":
        print("Использование: /mcp remove NAME")
        return

    name = parts[1]
    try:
        saved_path = remove_mcp_server(config_path, name)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return
    except OSError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return

    print(header_line("MCP"))
    print(status_line("Сервер удален", name, WARNING))
    print(status_line("Конфиг", str(saved_path), VALUE))


def clear_mcp_servers_from_command(config_path: Path) -> None:
    try:
        saved_path = clear_mcp_servers(config_path)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return
    except OSError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return

    print(header_line("MCP"))
    print(status_line("MCP", "не настроен", WARNING))
    print(status_line("Конфиг", str(saved_path), VALUE))
    print()
    print(indented_line("Все MCP-серверы удалены из config."))


def init_mock_mcp_config(config_path: Path) -> None:
    server = MCPServerConfig(
        name="mock-api",
        command=sys.executable,
        args=["-m", "code_agent_cli.mock_api_mcp_server"],
    )

    try:
        saved_path = add_mcp_server(config_path, server, overwrite=True)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return
    except OSError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return

    print(header_line("MCP"))
    print(status_line("Сервер подключен", "mock-api", SUCCESS))
    print(status_line("API", "http://jsonplaceholder.typicode.com", VALUE))
    print(status_line("Конфиг", str(saved_path), VALUE))
    print()
    print(indented_line("Проверить инструменты:"))
    print(indented_line("/mcp tools", level=2))
    print()
    print(indented_line("Вызвать инструмент:"))
    print(indented_line('/mcp call mock-api get_mock_user {"user_id": 1}', level=2))


def init_scheduler_mcp_config(config_path: Path) -> None:
    scheduler_env = {}
    scheduler_db = os.getenv("CODE_AGENT_SCHEDULER_DB")
    if scheduler_db:
        scheduler_env["CODE_AGENT_SCHEDULER_DB"] = scheduler_db

    server = MCPServerConfig(
        name="scheduler",
        command=sys.executable,
        args=["-m", "code_agent_cli.scheduler_mcp_server"],
        env=scheduler_env,
    )

    try:
        saved_path = add_mcp_server(config_path, server, overwrite=True)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return
    except OSError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return

    print(header_line("MCP"))
    print(status_line("Сервер подключен", "scheduler", SUCCESS))
    print(status_line("Хранилище", scheduler_db or "~/.code-agent-cli/scheduler.db", VALUE))
    print(status_line("Конфиг", str(saved_path), VALUE))
    print()
    print(indented_line("Проверить инструменты:"))
    print(indented_line("/mcp tools", level=2))
    print()
    print(indented_line("Создать reminder:"))
    print(
        indented_line(
            '/mcp remind "Проверить сводку" 2026-06-24T12:30:00Z',
            level=2,
        )
    )
    print()
    print(indented_line("Запустить due jobs вручную:"))
    print(indented_line("/mcp run_due", level=2))
    print()
    print(indented_line("Показать сводку:"))
    print(indented_line("/mcp summary", level=2))
    print()
    print(indented_line("Фоновый запуск:"))
    print(indented_line("scheduler-runner --watch --interval 60", level=2))


def init_pipeline_mcp_config(config_path: Path) -> None:
    pipeline_env = {}
    pipeline_dir = os.getenv("CODE_AGENT_PIPELINE_DIR")
    if pipeline_dir:
        pipeline_env["CODE_AGENT_PIPELINE_DIR"] = pipeline_dir

    server = MCPServerConfig(
        name="pipeline",
        command=sys.executable,
        args=["-m", "code_agent_cli.pipeline_mcp_server"],
        env=pipeline_env,
    )

    try:
        saved_path = add_mcp_server(config_path, server, overwrite=True)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return
    except OSError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return

    print(header_line("MCP"))
    print(status_line("Сервер подключен", "pipeline", SUCCESS))
    print(status_line("Вывод", pipeline_dir or "~/.code-agent-cli/pipeline", VALUE))
    print(status_line("Конфиг", str(saved_path), VALUE))
    print()
    print(indented_line("Проверить инструменты:"))
    print(indented_line("/mcp tools", level=2))
    print()
    print(indented_line("Запустить pipeline:"))
    print(indented_line('/mcp pipeline "latest MCP protocol news" mcp-summary.md', level=2))
    print()
    print(indented_line("Построить локальный индекс документов через Ollama:"))
    print(indented_line("/mcp index-docs .", level=2))
    print(indented_line("/mcp compare-chunking", level=2))


def init_orchestration_mcp_config(config_path: Path) -> None:
    try:
        existing_config = load_mcp_config_or_empty(config_path)
    except MCPConfigError as error:
        print(f"Ошибка MCP config: {error}", file=sys.stderr)
        return

    apple_payload = default_apple_mcp_config_payload()["mcpServers"]
    servers = [
        MCPServerConfig(
            name="apple-mcp",
            command=str(apple_payload["apple-mcp"]["command"]),
            args=list(apple_payload["apple-mcp"]["args"]),
        ),
        MCPServerConfig(
            name="cupertino",
            command=str(apple_payload["cupertino"]["command"]),
            args=list(apple_payload["cupertino"]["args"]),
        ),
        MCPServerConfig(
            name="scheduler",
            command=sys.executable,
            args=["-m", "code_agent_cli.scheduler_mcp_server"],
            env=scheduler_env_from_process(),
        ),
        MCPServerConfig(
            name="pipeline",
            command=sys.executable,
            args=["-m", "code_agent_cli.pipeline_mcp_server"],
            env=pipeline_env_from_process(),
        ),
    ]

    saved_path: Path | None = existing_config.path
    existing_names = {server.name for server in existing_config.servers}
    added_servers: list[MCPServerConfig] = []
    for server in servers:
        if server.name in existing_names:
            continue
        try:
            saved_path = add_mcp_server(config_path, server)
            added_servers.append(server)
        except MCPConfigError as error:
            print(f"Ошибка MCP config: {error}", file=sys.stderr)
            return
        except OSError as error:
            print(f"Ошибка MCP config: {error}", file=sys.stderr)
            return

    print(header_line("MCP"))
    print(status_line("Orchestration servers", "подключены", SUCCESS))
    print(status_line("Конфиг", str(saved_path or config_path), VALUE))
    for server in servers:
        state = "registered" if server.name in {item.name for item in added_servers} else "existing"
        print(status_line(server.name, state, VALUE))
    print()
    print(indented_line("Проверить tools:"))
    print(indented_line("/mcp tools", level=2))
    print(indented_line("Запустить длинный flow:"))
    print(indented_line('/mcp orchestrate "найди лучшие практики навигации SwiftUI в iOS через Apple/Cupertino MCP, сделай сводку, сохрани в заметки и поставь напоминание проверить завтра"', level=2))


def scheduler_env_from_process() -> dict[str, str]:
    scheduler_env = {}
    scheduler_db = os.getenv("CODE_AGENT_SCHEDULER_DB")
    if scheduler_db:
        scheduler_env["CODE_AGENT_SCHEDULER_DB"] = scheduler_db
    return scheduler_env


def pipeline_env_from_process() -> dict[str, str]:
    pipeline_env = {}
    pipeline_dir = os.getenv("CODE_AGENT_PIPELINE_DIR")
    if pipeline_dir:
        pipeline_env["CODE_AGENT_PIPELINE_DIR"] = pipeline_dir
    return pipeline_env


def call_mcp_tool_from_command(config_path: Path, argument: str) -> None:
    parts = argument.split(maxsplit=3)
    if len(parts) < 3 or parts[0] != "call":
        print('Использование: /mcp call SERVER TOOL {"param": "value"}')
        return

    server_name = parts[1]
    tool_name = parts[2]
    raw_arguments = parts[3] if len(parts) == 4 else "{}"

    try:
        tool_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        print(f"Ошибка MCP: аргументы tool должны быть JSON object: {error}", file=sys.stderr)
        return

    if not isinstance(tool_arguments, dict):
        print("Ошибка MCP: аргументы tool должны быть JSON object.", file=sys.stderr)
        return

    call_mcp_tool_from_values(config_path, server_name, tool_name, tool_arguments)


def call_scheduler_tool_from_short_command(
    config_path: Path,
    tool_name: str,
    tool_arguments: dict[str, Any],
) -> None:
    call_mcp_tool_from_values(config_path, "scheduler", tool_name, tool_arguments)


def call_pipeline_tool_from_short_command(
    config_path: Path,
    tool_name: str,
    tool_arguments: dict[str, Any],
) -> None:
    call_mcp_tool_from_values(config_path, "pipeline", tool_name, tool_arguments)


def call_mcp_tool_from_values(
    config_path: Path,
    server_name: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
) -> None:
    try:
        config = load_mcp_config(config_path)
    except MCPConfigError as error:
        print_mcp_config_missing(config_path, error if config_path.exists() else None)
        return

    server = find_mcp_server(config, server_name)
    if server is None:
        print(f"Ошибка MCP: server не найден: {server_name}", file=sys.stderr)
        if server_name == "scheduler":
            print("Подключите scheduler: /mcp init-scheduler")
        if server_name == "pipeline":
            print("Подключите pipeline: /mcp init-pipeline")
        return

    try:
        with loader(mcp_loader_label(server_name, tool_name)):
            result = asyncio.run(
                call_mcp_tool(
                    server.command,
                    server.args,
                    tool_name,
                    tool_arguments,
                    cwd=server.cwd,
                    env=server.env,
                    timeout=mcp_tool_timeout(server_name, tool_name),
                )
            )
    except FileNotFoundError:
        print(f"Ошибка MCP: команда server не найдена: {server.command}", file=sys.stderr)
        return
    except MCPConnectionError as error:
        print(f"Ошибка MCP: {error}", file=sys.stderr)
        return
    except Exception as error:
        print(f"Ошибка MCP: {error}", file=sys.stderr)
        return

    print_mcp_tool_call_result(server_name, tool_name, result)


def run_mcp_orchestration_from_command(agent: CodeAgent, argument: str) -> None:
    request = argument.strip()
    if not request:
        print('Использование: /mcp orchestrate "запрос для длинного MCP flow"')
        return

    try:
        with loader("Смотрю MCP tools"):
            config = load_mcp_config(default_mcp_config_file())
            catalog = load_mcp_orchestration_catalog(config)
    except MCPConfigError as error:
        print_mcp_config_missing(default_mcp_config_file(), error)
        return

    if not catalog.tools:
        print("Ошибка MCP orchestration: нет доступных tools.", file=sys.stderr)
        return

    try:
        with loader("Планирую MCP orchestration"):
            plan = agent.plan_mcp_orchestration(request, catalog.tools)
        plan = normalize_mcp_orchestration_plan(plan)
        print()
        print_mcp_orchestration_plan(plan)
        print()
        run_result = run_mcp_orchestration_plan(config, catalog, plan)
    except MissingAPIKeyError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return
    except Exception as error:
        print(f"Ошибка MCP orchestration: {error}", file=sys.stderr)
        return

    print_mcp_orchestration_result(run_result)


def load_mcp_orchestration_catalog(config: MCPConfig) -> MCPOrchestrationToolCatalog:
    preferred_servers = {"apple-mcp", "cupertino", "pipeline", "scheduler"}
    checks = [
        check_mcp_server(server, env_float("CODE_AGENT_MCP_TIMEOUT", 30.0))
        for server in config.servers
        if server.name in preferred_servers
    ]
    tools: list[MCPToolDescriptor] = []
    for check in checks:
        if not check.ok:
            continue
        for tool in check.tools:
            tools.append(
                MCPToolDescriptor(
                    server=check.server.name,
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.input_schema,
                )
            )
    return MCPOrchestrationToolCatalog(tools=tools, checks=checks)


def run_mcp_orchestration_plan(
    config: MCPConfig,
    catalog: MCPOrchestrationToolCatalog,
    plan: MCPOrchestrationPlan,
) -> MCPOrchestrationRunResult:
    validate_mcp_orchestration_plan(catalog, plan)
    step_results: list[MCPOrchestrationStepResult] = []

    for index, step in enumerate(plan.steps, start=1):
        server = find_mcp_server(config, step.server)
        if server is None:
            step_results.append(
                MCPOrchestrationStepResult(
                    index=index,
                    step=step,
                    arguments=step.arguments,
                    error=f"server не найден: {step.server}",
                )
            )
            break

        arguments = resolve_mcp_orchestration_arguments(step.arguments, step_results)
        try:
            with loader(f"Step {index}: {step.server}/{step.tool}"):
                result = asyncio.run(
                    call_mcp_tool(
                        server.command,
                        server.args,
                        step.tool,
                        arguments,
                        cwd=server.cwd,
                        env=server.env,
                        timeout=env_float("CODE_AGENT_MCP_TIMEOUT", 30.0),
                    )
                )
        except Exception as error:
            step_results.append(
                MCPOrchestrationStepResult(
                    index=index,
                    step=step,
                    arguments=arguments,
                    error=str(error),
                )
            )
            break

        step_results.append(
            MCPOrchestrationStepResult(
                index=index,
                step=step,
                arguments=arguments,
                result=result,
                error=compact_text(result.as_text(), max_length=500) if result.is_error else None,
            )
        )
        if result.is_error:
            break

    return MCPOrchestrationRunResult(plan=plan, steps=step_results)


def normalize_mcp_orchestration_plan(plan: MCPOrchestrationPlan) -> MCPOrchestrationPlan:
    steps: list[MCPOrchestrationStep] = []
    for step in plan.steps:
        steps.append(normalize_mcp_orchestration_step(step, steps))
    return MCPOrchestrationPlan(
        intent=plan.intent,
        steps=steps,
    )


def normalize_mcp_orchestration_step(
    step: MCPOrchestrationStep,
    previous_steps: list[MCPOrchestrationStep],
) -> MCPOrchestrationStep:
    if step.server == "pipeline" and step.tool == "summarize":
        previous_step = previous_steps[-1] if previous_steps else None
        if previous_step is not None and previous_step.server != "pipeline":
            arguments = dict(step.arguments)
            return MCPOrchestrationStep(
                server="pipeline",
                tool="summarize_text",
                arguments={
                    "query": str(arguments.get("query") or "Сделай краткую сводку результата предыдущего MCP tool"),
                    "content": "$previous_text",
                },
                reason=step.reason or "Суммаризация текста из предыдущего MCP tool",
            )

    if step.server == "pipeline" and step.tool == "save":
        arguments = dict(step.arguments)
        content = arguments.get("content")
        previous_step = previous_steps[-1] if previous_steps else None
        if (
            previous_step is not None
            and previous_step.server == "pipeline"
            and previous_step.tool in {"summarize", "summarize_text"}
            and isinstance(content, str)
            and re.fullmatch(r"\$steps\[\d+]", content)
        ):
            arguments["content"] = f"{content}.summary"
            return MCPOrchestrationStep(
                server=step.server,
                tool=step.tool,
                arguments=arguments,
                reason=step.reason,
            )

    if step.server != "cupertino" or step.tool != "search":
        return step

    arguments = dict(step.arguments)
    query = str(arguments.get("query") or "")
    lowered = query.lower()
    if "swiftui" in lowered and any(marker in lowered for marker in ("navigation", "navigationstack", "навигац")):
        arguments["query"] = (
            "SwiftUI NavigationStack NavigationSplitView tab navigation "
            "robust navigation best practices"
        )
        arguments["source"] = "all"
        arguments["limit"] = int(arguments.get("limit") or 10)
        arguments.pop("framework", None)

    return MCPOrchestrationStep(
        server=step.server,
        tool=step.tool,
        arguments=arguments,
        reason=step.reason,
    )


def validate_mcp_orchestration_plan(
    catalog: MCPOrchestrationToolCatalog,
    plan: MCPOrchestrationPlan,
) -> None:
    if len(plan.steps) > 6:
        raise ValueError("план содержит больше 6 шагов")
    allowed = {(tool.server, tool.name) for tool in catalog.tools}
    for step in plan.steps:
        if (step.server, step.tool) not in allowed:
            raise ValueError(f"tool не найден в catalog: {step.server}/{step.tool}")
        if not isinstance(step.arguments, dict):
            raise ValueError(f"arguments должны быть object: {step.server}/{step.tool}")


def resolve_mcp_orchestration_arguments(
    value: Any,
    step_results: list[MCPOrchestrationStepResult],
) -> Any:
    if isinstance(value, dict):
        return {
            key: resolve_mcp_orchestration_arguments(item, step_results)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_mcp_orchestration_arguments(item, step_results) for item in value]
    if isinstance(value, str):
        return resolve_mcp_orchestration_reference(value, step_results)
    return value


def resolve_mcp_orchestration_reference(
    value: str,
    step_results: list[MCPOrchestrationStepResult],
) -> Any:
    if value == "$tomorrow_09_utc":
        return tomorrow_utc_at(9).isoformat().replace("+00:00", "Z")
    if value == "$previous_text":
        return step_results[-1].result.as_text() if step_results and step_results[-1].result else ""
    match = re.fullmatch(r"\$steps\[(\d+)](?:\.(.+))?", value)
    if not match:
        return value

    step_index = int(match.group(1))
    if step_index >= len(step_results) and 1 <= step_index <= len(step_results):
        step_index -= 1
    if step_index < 0 or step_index >= len(step_results):
        return ""
    result = step_results[step_index].result
    if result is None:
        return ""
    path = match.group(2)
    if not path:
        return result.as_text()
    payload: Any = result.structured_content
    if payload is None:
        try:
            payload = json.loads(result.as_text())
        except json.JSONDecodeError:
            return result.as_text()
    for part in path.split("."):
        if isinstance(payload, dict):
            payload = payload.get(part)
        elif isinstance(payload, list) and part.isdigit():
            index = int(part)
            payload = payload[index] if 0 <= index < len(payload) else None
        else:
            payload = None
        if payload is None:
            return ""
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return payload


def tomorrow_utc_at(hour: int) -> datetime:
    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
    return datetime.combine(tomorrow, datetime_time(hour=hour, tzinfo=timezone.utc))


def mcp_loader_label(server_name: str, tool_name: str) -> str:
    if server_name == "pipeline" and tool_name == "run":
        return "Выполняю MCP pipeline"
    if server_name == "pipeline" and tool_name == "index_documents":
        return "Индексирую документы через Ollama"
    if server_name == "pipeline":
        return f"Выполняю pipeline/{tool_name}"
    return f"Выполняю MCP {server_name}/{tool_name}"


def mcp_tool_timeout(server_name: str, tool_name: str) -> float:
    long_pipeline_tools = {"index_documents", "rag_answer", "rag_compare", "rag_eval"}
    default_timeout = 300.0 if server_name == "pipeline" and tool_name in long_pipeline_tools else 30.0
    return env_float("CODE_AGENT_MCP_TIMEOUT", default_timeout)


def handle_scheduler_short_command(config_path: Path, command: str, argument: str) -> bool:
    if command == "remind":
        reminder = parse_scheduler_remind_arguments(argument)
        if reminder is None:
            print('Использование: /mcp remind "Текст напоминания" 2026-06-24T12:30:00Z')
            return True
        call_scheduler_tool_from_short_command(config_path, "remind", reminder)
        return True

    if command == "every":
        periodic = parse_scheduler_every_arguments(argument)
        if periodic is None:
            print('Использование: /mcp every "Daily summary" 1440 "Собрать краткую сводку"')
            return True
        call_scheduler_tool_from_short_command(config_path, "every", periodic)
        return True

    if command == "jobs":
        call_scheduler_tool_from_short_command(config_path, "jobs", {})
        return True

    if command == "run_due":
        tool_arguments = parse_optional_limit_argument(argument)
        if tool_arguments is None:
            print("Использование: /mcp run_due [LIMIT]")
            return True
        call_scheduler_tool_from_short_command(config_path, "run_due", tool_arguments)
        return True

    if command == "summary":
        tool_arguments = parse_optional_limit_argument(argument)
        if tool_arguments is None:
            print("Использование: /mcp summary [LIMIT]")
            return True
        call_scheduler_tool_from_short_command(config_path, "summary", tool_arguments)
        return True

    if command in {"clear-scheduler", "scheduler-clear", "clear_scheduler"}:
        call_scheduler_tool_from_short_command(config_path, "clear", {})
        return True

    if command == "health":
        call_scheduler_tool_from_short_command(config_path, "health", {})
        return True

    return False


def handle_pipeline_short_command(config_path: Path, command: str, argument: str) -> bool:
    if command == "rag-search":
        question = parse_required_text_argument("rag-search", argument)
        if question is None:
            return True
        call_pipeline_tool_from_short_command(config_path, "rag_search", {"question": question})
        return True

    if command == "rag-answer":
        question = parse_required_text_argument("rag-answer", argument)
        if question is None:
            return True
        call_pipeline_tool_from_short_command(config_path, "rag_answer", {"question": question, "use_rag": True})
        return True

    if command == "rag-compare":
        question = parse_required_text_argument("rag-compare", argument)
        if question is None:
            return True
        call_pipeline_tool_from_short_command(config_path, "rag_compare", {"question": question})
        return True

    if command == "rag-eval-questions":
        call_pipeline_tool_from_short_command(config_path, "rag_eval_questions", {})
        return True

    if command == "rag-eval":
        arguments: dict[str, Any] = {"run_answers": True}
        if argument.strip():
            try:
                max_questions = int(argument.strip())
            except ValueError:
                print("Использование: /mcp rag-eval [MAX_QUESTIONS]")
                return True
            arguments["max_questions"] = max_questions
        call_pipeline_tool_from_short_command(config_path, "rag_eval", arguments)
        return True

    if command == "index-docs":
        path = argument.strip() or "."
        call_pipeline_tool_from_short_command(
            config_path,
            "index_documents",
            {
                "path": path,
            },
        )
        return True

    if command in {"index-status", "docs-index-status"}:
        call_pipeline_tool_from_short_command(config_path, "index_status", {})
        return True

    if command in {"compare-chunking", "chunking"}:
        call_pipeline_tool_from_short_command(config_path, "compare_chunking", {})
        return True

    if command != "pipeline":
        return False

    try:
        parts = shlex.split(argument)
    except ValueError as error:
        print(f"Ошибка pipeline: {error}", file=sys.stderr)
        return True

    if len(parts) != 2:
        print('Использование: /mcp pipeline "search query" result.md')
        return True

    call_pipeline_tool_from_short_command(
        config_path,
        "run",
        {
            "query": parts[0],
            "filename": parts[1],
        },
    )
    return True


def parse_required_text_argument(command: str, argument: str) -> str | None:
    try:
        parts = shlex.split(argument)
    except ValueError as error:
        print(f"Ошибка {command}: {error}", file=sys.stderr)
        return None
    text = " ".join(parts).strip()
    if not text:
        print(f'Использование: /mcp {command} "вопрос"')
        return None
    return text


def parse_scheduler_remind_arguments(argument: str) -> dict[str, Any] | None:
    try:
        parts = shlex.split(argument)
    except ValueError as error:
        print(f"Ошибка scheduler: {error}", file=sys.stderr)
        return None
    if len(parts) < 2:
        return None
    return {
        "text": " ".join(parts[:-1]).strip(),
        "run_at": parts[-1],
    }


def parse_scheduler_every_arguments(argument: str) -> dict[str, Any] | None:
    try:
        parts = shlex.split(argument)
    except ValueError as error:
        print(f"Ошибка scheduler: {error}", file=sys.stderr)
        return None
    if len(parts) < 3:
        return None
    try:
        interval_minutes = int(parts[1])
    except ValueError:
        return None
    return {
        "title": parts[0],
        "interval_minutes": interval_minutes,
        "summary_text": " ".join(parts[2:]).strip(),
    }


def parse_optional_limit_argument(argument: str) -> dict[str, Any] | None:
    if not argument:
        return {}
    try:
        limit = int(argument.strip())
    except ValueError:
        return None
    if limit < 1:
        return None
    return {"limit": limit}


def warn_bare_scheduler_tool(text: str) -> bool:
    parts = text.split(maxsplit=1)
    if len(parts) != 1:
        return False
    command = parts[0].lower()
    if command not in {"health", "remind", "every", "jobs", "delete", "run_due", "summary"}:
        return False

    print(header_line("Scheduler"))
    print(status_line("Команда", command, WARNING))
    print(status_line("Статус", "это MCP tool, не обычный prompt", WARNING))
    print()
    print(indented_line("Используйте короткие команды через /mcp:"))
    print(indented_line('/mcp remind "Проверить планировщик" 2026-06-24T12:30:00Z', level=2))
    print(indented_line("/mcp run_due", level=2))
    print(indented_line("/mcp summary", level=2))
    return True


def find_mcp_server(config: MCPConfig, name: str) -> MCPServerConfig | None:
    for server in config.servers:
        if server.name == name:
            return server
    return None


def print_mcp_tool_call_result(
    server_name: str,
    tool_name: str,
    result: MCPToolCallResult,
) -> None:
    print(header_line("MCP call"))
    print(status_line("Server", server_name, VALUE))
    print(status_line("Tool", tool_name, VALUE))
    print(status_line("Status", "Error" if result.is_error else "OK", ERROR if result.is_error else SUCCESS))
    print()
    if print_pipeline_tool_call_result(server_name, tool_name, result):
        return
    if print_scheduler_tool_call_result(server_name, tool_name, result):
        return
    print(result.as_text())


def print_pipeline_tool_call_result(
    server_name: str,
    tool_name: str,
    result: MCPToolCallResult,
) -> bool:
    if server_name != "pipeline" or result.is_error:
        return False

    payload = parse_mcp_json_result(result)
    if payload is None:
        return False

    if tool_name == "search":
        print_pipeline_search_result(payload)
        return True
    if tool_name == "summarize":
        print_pipeline_summary_result(payload)
        return True
    if tool_name == "save":
        print_pipeline_save_result(payload)
        return True
    if tool_name == "run":
        print_pipeline_run_result(payload)
        return True
    if tool_name == "index_documents":
        print_document_index_result(payload)
        return True
    if tool_name == "index_status":
        print_document_index_status(payload)
        return True
    if tool_name == "compare_chunking":
        print_document_chunking_comparison(payload)
        return True
    if tool_name == "rag_search":
        print_rag_search_result(payload)
        return True
    if tool_name == "rag_answer":
        print_rag_answer_result(payload)
        return True
    if tool_name == "rag_compare":
        print_rag_compare_result(payload)
        return True
    if tool_name == "rag_eval_questions":
        print_rag_eval_questions(payload)
        return True
    if tool_name == "rag_eval":
        print_rag_eval_result(payload)
        return True
    if tool_name == "health":
        print(status_line("Состояние", str(payload.get("status", "")), SUCCESS))
        print(status_line("Output dir", str(payload.get("output_dir", "")), VALUE))
        return True

    return False


def print_pipeline_search_result(payload: dict[str, Any]) -> None:
    print(colorize("Pipeline step 1: search", BOLD + ACCENT))
    print(status_line("Query", str(payload.get("query", "")), VALUE))
    print(status_line("Results", str(payload.get("count", 0)), SUCCESS))
    for result in ensure_list(payload.get("results"))[:3]:
        if isinstance(result, dict):
            print(status_line(str(result.get("title", "")), str(result.get("url", "")), VALUE))


def print_pipeline_summary_result(payload: dict[str, Any]) -> None:
    print(colorize("Pipeline step 2: summarize", BOLD + ACCENT))
    print(status_line("Query", str(payload.get("query", "")), VALUE))
    print(status_line("Items used", str(payload.get("items_used", 0)), SUCCESS))
    print(status_line("Model", str(payload.get("model", "")), VALUE))
    usage = payload.get("usage")
    if isinstance(usage, dict) and usage.get("total_tokens") is not None:
        print(status_line("Tokens", str(usage["total_tokens"]), VALUE))
    print_multiline_value("Summary", str(payload.get("summary", "")))


def print_pipeline_save_result(payload: dict[str, Any]) -> None:
    saved = bool(payload.get("saved"))
    print(colorize("Pipeline step 3: save", BOLD + ACCENT))
    print(status_line("Saved", "yes" if saved else "no", SUCCESS if saved else ERROR))
    print(status_line("Path", str(payload.get("path", "")), VALUE))
    print(status_line("Bytes", str(payload.get("bytes", 0)), VALUE))


def print_pipeline_run_result(payload: dict[str, Any]) -> None:
    search_payload = payload.get("search") if isinstance(payload.get("search"), dict) else {}
    summary_payload = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    save_payload = payload.get("save") if isinstance(payload.get("save"), dict) else {}

    print(colorize("Автоматический MCP pipeline", BOLD + ACCENT))
    print(status_line("Цепочка", str(payload.get("pipeline", "")), SUCCESS))
    print(status_line("Query", str(payload.get("query", "")), VALUE))
    print()
    print_pipeline_search_result(search_payload)
    print()
    print_pipeline_summary_result(summary_payload)
    print()
    print_pipeline_save_result(save_payload)


def print_document_index_result(payload: dict[str, Any]) -> None:
    documents = payload.get("documents") if isinstance(payload.get("documents"), dict) else {}
    chunks = payload.get("chunks") if isinstance(payload.get("chunks"), dict) else {}
    embedding = payload.get("embedding") if isinstance(payload.get("embedding"), dict) else {}

    print(colorize("Document index", BOLD + ACCENT))
    print(status_line("Status", "OK", SUCCESS))
    print(status_line("Root", str(payload.get("root", "")), VALUE))
    print(status_line("SQLite", str(payload.get("db_path", "")), VALUE))
    print(status_line("Report", str(payload.get("report_path", "")), VALUE))
    print(status_line("Embedding", f"{embedding.get('provider', '')}/{embedding.get('model', '')}", VALUE))
    print(status_line("Documents", str(documents.get("count", 0)), SUCCESS))
    print(status_line("Estimated pages", str(documents.get("estimated_pages", 0)), VALUE))
    print(status_line("Chunks", str(chunks.get("total", 0)), SUCCESS))
    print(status_line("Chunk target", f"{chunks.get('target_size_tokens', '')} tokens", VALUE))
    print(status_line("Overlap", f"{chunks.get('overlap_tokens', '')} tokens", VALUE))
    print()
    print_document_chunking_comparison(payload)


def print_document_index_status(payload: dict[str, Any]) -> None:
    print(colorize("Document index status", BOLD + ACCENT))
    if not payload.get("exists"):
        print(status_line("Status", "index not found", WARNING))
        print(status_line("SQLite", str(payload.get("db_path", "")), VALUE))
        print(indented_line("Создать индекс: /mcp index-docs .", level=1))
        return

    print(status_line("Status", "OK", SUCCESS))
    print(status_line("Root", str(payload.get("root", "")), VALUE))
    print(status_line("SQLite", str(payload.get("db_path", "")), VALUE))
    print(status_line("Report", str(payload.get("report_path", "")), VALUE))
    print(status_line("Model", str(payload.get("model", "")), VALUE))
    print(status_line("Sources", str(payload.get("sources", 0)), SUCCESS))
    print(status_line("Chunks", str(payload.get("chunks", 0)), SUCCESS))
    by_strategy = payload.get("by_strategy")
    if isinstance(by_strategy, dict):
        for strategy, count in by_strategy.items():
            print(status_line(f"Strategy {strategy}", str(count), VALUE))


def print_document_chunking_comparison(payload: dict[str, Any]) -> None:
    chunks = payload.get("chunks") if isinstance(payload.get("chunks"), dict) else {}
    strategies = chunks.get("strategies") if isinstance(chunks.get("strategies"), dict) else {}
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}

    print(colorize("Chunking comparison", BOLD + ACCENT))
    if not strategies:
        print(status_line("Status", "нет данных сравнения", WARNING))
        return

    for name in ("fixed", "structural"):
        stats = strategies.get(name)
        if not isinstance(stats, dict):
            continue
        print(status_line(f"{name} chunks", str(stats.get("chunks", 0)), SUCCESS))
        print(status_line(f"{name} avg tokens", str(stats.get("avg_tokens", 0)), VALUE))
        print(status_line(f"{name} sections", str(stats.get("sections", 0)), VALUE))

    if comparison:
        print()
        for key in ("fixed", "structural", "chunk_count_delta", "section_coverage"):
            value = comparison.get(key)
            if value:
                print_multiline_value(key, str(value))


def print_rag_search_result(payload: dict[str, Any]) -> None:
    print(colorize("RAG search", BOLD + ACCENT))
    print(status_line("Question", str(payload.get("question", "")), VALUE))
    print(status_line("Mode", str(payload.get("mode", "")), SUCCESS))
    print(status_line("Grounding", str(payload.get("grounding_status", "")), SUCCESS))
    print(status_line("Best similarity", str(payload.get("best_similarity", "")), VALUE))
    rewritten = str(payload.get("rewritten_question", ""))
    if rewritten:
        print_multiline_value("Rewrite", rewritten)
    print(status_line("Top K", str(payload.get("top_k", 0)), VALUE))
    print(status_line("Candidate K", str(payload.get("candidate_k", 0)), VALUE))
    if payload.get("min_similarity") is not None:
        print(status_line("Min similarity", str(payload.get("min_similarity", "")), VALUE))
        print(status_line("After filter", str(payload.get("candidates_after_filter", 0)), SUCCESS))
        print(status_line("Final chunks", str(payload.get("top_k_after_filter", 0)), SUCCESS))
        print(status_line("Filtered out", str(payload.get("filtered_out", 0)), VALUE))
    chunks = ensure_list(payload.get("chunks"))
    print(status_line("Chunks", str(len(chunks)), SUCCESS))
    for index, chunk in enumerate(chunks[:5], start=1):
        if not isinstance(chunk, dict):
            continue
        print()
        print(status_line(f"Chunk {index}", str(chunk.get("chunk_id", "")), SUCCESS))
        print(status_line("Source", str(chunk.get("source", "")), VALUE))
        print(status_line("Section", str(chunk.get("section", "")), VALUE))
        print(status_line("Score", str(chunk.get("score", "")), VALUE))
        if chunk.get("similarity") is not None:
            print(status_line("Similarity", str(chunk.get("similarity", "")), VALUE))
        print_multiline_value("Preview", str(chunk.get("preview", "")))


def print_rag_answer_result(payload: dict[str, Any]) -> None:
    print(colorize("RAG answer", BOLD + ACCENT))
    print(status_line("Mode", str(payload.get("mode", "")), SUCCESS))
    print(status_line("Grounding", str(payload.get("grounding_status", "")), SUCCESS))
    print(status_line("Best similarity", str(payload.get("best_similarity", "")), VALUE))
    print(status_line("Question", str(payload.get("question", "")), VALUE))
    print(status_line("Model", str(payload.get("model", "")), VALUE))
    print_multiline_value("Answer", str(payload.get("answer", "")))
    print_rag_sources(payload.get("sources"))
    print_rag_quotes(payload.get("quotes"))


def print_rag_compare_result(payload: dict[str, Any]) -> None:
    print(colorize("RAG comparison", BOLD + ACCENT))
    print(status_line("Question", str(payload.get("question", "")), VALUE))
    without_rag = payload.get("without_rag") if isinstance(payload.get("without_rag"), dict) else {}
    baseline_rag = payload.get("baseline_rag") if isinstance(payload.get("baseline_rag"), dict) else {}
    with_rag = payload.get("with_rag") if isinstance(payload.get("with_rag"), dict) else {}
    print()
    print_multiline_value("Without RAG", str(without_rag.get("answer", "")))
    print()
    print_multiline_value("Baseline RAG", str(baseline_rag.get("answer", "")))
    print_rag_sources(baseline_rag.get("sources"))
    print_rag_quotes(baseline_rag.get("quotes"))
    print()
    print_multiline_value("Enhanced RAG", str(with_rag.get("answer", "")))
    print_rag_sources(with_rag.get("sources"))
    print_rag_quotes(with_rag.get("quotes"))
    retrieval_modes = payload.get("retrieval_modes")
    if isinstance(retrieval_modes, dict):
        print()
        print(colorize("Retrieval modes", BOLD + ACCENT))
        for name in ("baseline", "enhanced"):
            retrieval = retrieval_modes.get(name)
            if not isinstance(retrieval, dict):
                continue
            print(
                status_line(
                    name,
                    (
                        f"candidate_k={retrieval.get('candidate_k', '')}, "
                        f"after_filter={retrieval.get('candidates_after_filter', '')}, "
                        f"final={retrieval.get('top_k_after_filter', '')}, "
                        f"filtered_out={retrieval.get('filtered_out', '')}"
                    ),
                    VALUE,
                )
            )
    note = str(payload.get("quality_note", ""))
    if note:
        print()
        print_multiline_value("Quality note", note)


def print_rag_eval_questions(payload: dict[str, Any]) -> None:
    print(colorize("RAG eval questions", BOLD + ACCENT))
    print(status_line("Questions", str(payload.get("count", 0)), SUCCESS))
    for index, item in enumerate(ensure_list(payload.get("questions")), start=1):
        if not isinstance(item, dict):
            continue
        print()
        print(status_line(f"Question {index}", str(item.get("question", "")), SUCCESS))
        print_multiline_value("Expected", str(item.get("expected", "")))
        sources = item.get("expected_sources")
        if isinstance(sources, list):
            print(status_line("Expected sources", ", ".join(str(source) for source in sources), VALUE))


def print_rag_eval_result(payload: dict[str, Any]) -> None:
    print(colorize("RAG eval", BOLD + ACCENT))
    print(status_line("Questions", str(payload.get("questions", 0)), SUCCESS))
    print(status_line("Top K", str(payload.get("top_k", 0)), VALUE))
    print(status_line("Candidate K", str(payload.get("candidate_k", 0)), VALUE))
    print(status_line("Min similarity", str(payload.get("min_similarity", "")), VALUE))
    print(status_line("Answers", "run" if payload.get("run_answers") else "not run", VALUE))
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key, value in summary.items():
            print(status_line(str(key), str(value), VALUE))
    for index, item in enumerate(ensure_list(payload.get("results")), start=1):
        if not isinstance(item, dict):
            continue
        print()
        print(status_line(f"Question {index}", str(item.get("question", "")), SUCCESS))
        print_multiline_value("Expected", str(item.get("expected", "")))
        source_hits = item.get("source_hits")
        if isinstance(source_hits, dict):
            rendered_hits = ", ".join(
                f"{source}={'ok' if matched else 'miss'}"
                for source, matched in source_hits.items()
            )
            print(status_line("Enhanced source hits", rendered_hits, VALUE))
        baseline_source_hits = item.get("baseline_source_hits")
        if isinstance(baseline_source_hits, dict):
            rendered_hits = ", ".join(
                f"{source}={'ok' if matched else 'miss'}"
                for source, matched in baseline_source_hits.items()
            )
            print(status_line("Baseline source hits", rendered_hits, VALUE))
        if item.get("with_rag"):
            print(status_line("Enhanced grounding", str(item.get("with_rag_grounding_status", "")), VALUE))
            print(status_line("Has sources", "yes" if item.get("with_rag_has_sources") else "no", VALUE))
            print(status_line("Has quotes", "yes" if item.get("with_rag_has_quotes") else "no", VALUE))
            alignment = item.get("with_rag_answer_quote_alignment")
            if isinstance(alignment, dict):
                print(
                    status_line(
                        "Answer/quote alignment",
                        f"{'ok' if alignment.get('aligned') else 'weak'} score={alignment.get('score', '')}",
                        VALUE,
                    )
                )
            print_multiline_value("Enhanced RAG", str(item.get("with_rag", "")))
        if item.get("baseline_rag"):
            print_multiline_value("Baseline RAG", str(item.get("baseline_rag", "")))
        if item.get("without_rag"):
            print_multiline_value("Without RAG", str(item.get("without_rag", "")))


def print_rag_sources(value: Any) -> None:
    sources = ensure_list(value)
    if not sources:
        return
    print()
    print(status_line("Sources", str(len(sources)), SUCCESS))
    for source in sources[:8]:
        if not isinstance(source, dict):
            continue
        print(
            status_line(
                str(source.get("source", "")),
                (
                    f"{source.get('section', '')} · {source.get('chunk_id', '')} · "
                    f"score={source.get('score', '')} · similarity={source.get('similarity', '')}"
                ),
                VALUE,
            )
        )


def print_rag_quotes(value: Any) -> None:
    quotes = ensure_list(value)
    if not quotes:
        return
    print()
    print(status_line("Quotes", str(len(quotes)), SUCCESS))
    for quote in quotes[:8]:
        if not isinstance(quote, dict):
            continue
        print(
            status_line(
                str(quote.get("source", "")),
                f"{quote.get('section', '')} · {quote.get('chunk_id', '')}",
                VALUE,
            )
        )
        print_multiline_value("Quote", str(quote.get("quote", "")))


def print_multiline_value(label: str, text: str) -> None:
    print(status_line(label, "", SUCCESS).rstrip())
    rendered = render_answer(text.strip())
    if not rendered:
        print(indented_line("пусто", level=1))
        return
    for line in rendered:
        print(f"  {line}")


def print_mcp_orchestration_result(run_result: MCPOrchestrationRunResult) -> None:
    print(header_line("MCP Orchestration"))
    print(status_line("Intent", run_result.plan.intent, SUCCESS))
    servers = " -> ".join(f"{step.step.server}/{step.step.tool}" for step in run_result.steps)
    print(status_line("Flow", servers or "нет шагов", VALUE))
    print()
    for step_result in run_result.steps:
        print(
            colorize(
                f"Step {step_result.index}: {step_result.step.server}/{step_result.step.tool}",
                BOLD + ACCENT,
            )
        )
        print(status_line("Status", "OK" if step_result.ok else "Error", SUCCESS if step_result.ok else ERROR))
        if step_result.error:
            print(status_line("Error", step_result.error, ERROR))
        elif step_result.result is not None:
            print_mcp_orchestration_step_payload(step_result)
        print()


def print_mcp_orchestration_plan(plan: MCPOrchestrationPlan) -> None:
    print(header_line("MCP Orchestration plan"))
    print(status_line("Intent", plan.intent, SUCCESS))
    flow = " -> ".join(f"{step.server}/{step.tool}" for step in plan.steps)
    print(status_line("Flow", flow or "нет шагов", VALUE))
    for index, step in enumerate(plan.steps, start=1):
        print(status_line(f"Step {index}", f"{step.server}/{step.tool}", VALUE))


def print_mcp_orchestration_step_payload(step_result: MCPOrchestrationStepResult) -> None:
    result = step_result.result
    if result is None:
        return
    payload = parse_mcp_json_result(result)
    if step_result.step.server == "pipeline" and step_result.step.tool == "run" and payload:
        save_payload = payload.get("save") if isinstance(payload.get("save"), dict) else {}
        summary_payload = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        print(status_line("Saved", str(save_payload.get("path", "")), SUCCESS))
        print_multiline_value("Summary", str(summary_payload.get("summary", "")))
        return
    if step_result.step.server == "pipeline" and step_result.step.tool in {"summarize", "summarize_text"} and payload:
        print_multiline_value("Summary", str(payload.get("summary", "")))
        return
    if step_result.step.server == "pipeline" and step_result.step.tool == "save" and payload:
        print(status_line("Saved", str(payload.get("path", "")), SUCCESS))
        return
    if step_result.step.server == "cupertino" and step_result.step.tool == "search":
        print_mcp_orchestration_search_result(result.as_text())
        return
    if step_result.step.server == "scheduler" and step_result.step.tool == "remind" and payload:
        print(status_line("Reminder", str(payload.get("title") or payload.get("text") or "created"), SUCCESS))
        print(status_line("Next run", str(payload.get("next_run_at", "")), VALUE))
        return
    if step_result.step.server == "scheduler" and step_result.step.tool == "summary" and payload:
        active_jobs = payload.get("active_jobs", 0)
        recent_runs = len(ensure_list(payload.get("recent_runs")))
        print(status_line("Scheduler summary", f"{active_jobs} active jobs, {recent_runs} recent runs", SUCCESS))
        print(status_line("Reminder", "saved and visible in scheduler", SUCCESS))
        return

    text = result.as_text()
    if len(text) > 1200:
        text = f"{text[:1200].rstrip()}\n..."
    print_multiline_value("Result", text)


def print_mcp_orchestration_search_result(text: str) -> None:
    count_match = re.search(r"(?:Total:\s*)?(\d+)\s+results?", text, re.IGNORECASE)
    found = count_match.group(1) if count_match else "some"
    print(status_line("Search", f"{found} results from Cupertino", SUCCESS))

    interesting_lines = [
        line.strip(" -")
        for line in text.splitlines()
        if line.strip().startswith("- **")
    ][:3]
    if interesting_lines:
        print(status_line("Top", "; ".join(interesting_lines), VALUE))


def compact_text(text: str, *, max_length: int) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length].rstrip()}..."


def print_compact_json_value(label: str, value: Any, *, max_length: int = 280) -> None:
    compacted = compact_large_json_values(value, max_length=max_length)
    text = json.dumps(compacted, ensure_ascii=False, indent=2)
    print(status_line(label, "", SUCCESS).rstrip())
    for line in text.splitlines():
        wrapped_lines = textwrap.wrap(
            line,
            width=terminal_text_width(),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [line]
        for wrapped_line in wrapped_lines:
            print(colorize(f"  {wrapped_line}", VALUE))


def compact_large_json_values(value: Any, *, max_length: int) -> Any:
    if isinstance(value, dict):
        return {
            key: compact_large_json_values(item, max_length=max_length)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [compact_large_json_values(item, max_length=max_length) for item in value]
    if isinstance(value, str) and len(value) > max_length:
        return f"{value[:max_length].rstrip()}..."
    return value


def print_scheduler_tool_call_result(
    server_name: str,
    tool_name: str,
    result: MCPToolCallResult,
) -> bool:
    if server_name != "scheduler" or result.is_error:
        return False

    payload = parse_mcp_json_result(result)
    if payload is None:
        return False

    if tool_name in {"remind", "every"}:
        print_scheduler_created_job(tool_name, payload)
        return True
    if tool_name == "jobs":
        print_scheduler_jobs(payload)
        return True
    if tool_name == "run_due":
        print_scheduler_due_result(payload)
        return True
    if tool_name == "summary":
        print_scheduler_summary(payload)
        return True
    if tool_name == "delete":
        print(status_line("Job", str(payload.get("job_id", "")), VALUE))
        deleted = bool(payload.get("deleted"))
        print(status_line("Удален", "да" if deleted else "нет", SUCCESS if deleted else WARNING))
        return True
    if tool_name == "clear":
        print(status_line("Scheduler", "cleared", SUCCESS))
        print(status_line("Jobs deleted", str(payload.get("jobs_deleted", 0)), VALUE))
        print(status_line("Runs deleted", str(payload.get("runs_deleted", 0)), VALUE))
        return True
    if tool_name == "health":
        print(status_line("Состояние", str(payload.get("status", "")), SUCCESS))
        print(status_line("SQLite", str(payload.get("database", "")), VALUE))
        return True

    return False


def parse_mcp_json_result(result: MCPToolCallResult) -> dict[str, Any] | None:
    try:
        payload = json.loads(result.as_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def print_scheduler_created_job(tool_name: str, payload: dict[str, Any]) -> None:
    kind = "Reminder" if tool_name == "remind" else "Periodic summary"
    print(colorize(f"{kind} создан", BOLD + SUCCESS))
    print(status_line("Job ID", str(payload.get("id", "")), VALUE))
    print(status_line("Название", str(payload.get("title", "")), VALUE))
    print(status_line("Тип", str(payload.get("kind", "")), VALUE))
    print(status_line("Следующий запуск", str(payload.get("next_run_at", "")), WARNING))
    interval_seconds = payload.get("interval_seconds")
    if isinstance(interval_seconds, int):
        print(status_line("Интервал", format_seconds(interval_seconds), VALUE))
    print(status_line("Сохранено", "SQLite jobs", SUCCESS))


def print_scheduler_jobs(payload: dict[str, Any]) -> None:
    jobs = ensure_list(payload.get("jobs"))
    print(colorize("Список задач", BOLD + ACCENT))
    print(status_line("Активных/найдено", str(payload.get("count", len(jobs))), VALUE))
    if not jobs:
        print(status_line("Очередь", "пусто", WARNING))
        return

    print()
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            continue
        enabled = bool(job.get("enabled"))
        state_color = SUCCESS if enabled else WARNING
        print(command_line(f"{index}. {job.get('title', '')}"))
        print(status_line("  Статус", "active" if enabled else "disabled", state_color))
        print(status_line("  Тип", str(job.get("kind", "")), VALUE))
        print(status_line("  Следующий запуск", str(job.get("next_run_at", "")), WARNING))


def print_scheduler_due_result(payload: dict[str, Any]) -> None:
    runs = ensure_list(payload.get("runs"))
    success_count = count_runs_by_status(runs, "success")
    error_count = len(runs) - success_count

    print(colorize("Выполнение расписания", BOLD + ACCENT))
    print(status_line("Проверено в", str(payload.get("checked_at", "")), VALUE))
    print(status_line("Due jobs", str(payload.get("due_jobs", len(runs))), WARNING if runs else VALUE))
    print(status_line("Успешно", str(success_count), SUCCESS))
    print(status_line("Ошибок", str(error_count), ERROR if error_count else SUCCESS))
    print(status_line("Сохранено", "SQLite job_runs", SUCCESS if runs else WARNING))

    if not runs:
        print()
        print(status_line("Итог", "нет задач, срок которых наступил", WARNING))
        return

    print()
    print(colorize("Запуски", BOLD + ACCENT_SOFT))
    for run in runs:
        if isinstance(run, dict):
            print_scheduler_run_line(run)


def print_scheduler_summary(payload: dict[str, Any]) -> None:
    recent_runs = ensure_list(payload.get("recent_runs"))
    next_runs = ensure_list(payload.get("next_runs"))
    failed_runs = int(payload.get("failed_runs") or 0)
    last_run = first_dict(recent_runs)

    print(colorize("Сводка планировщика", BOLD + ACCENT))
    print(status_line("Сформирована", str(payload.get("generated_at", "")), VALUE))
    print(status_line("Активных задач", str(payload.get("active_jobs", 0)), VALUE))
    print(status_line("Последних запусков", str(len(recent_runs)), VALUE))
    print(status_line("Ошибок", str(failed_runs), ERROR if failed_runs else SUCCESS))
    print(status_line("Данные", "SQLite jobs + job_runs", SUCCESS))

    print()
    if last_run is None:
        print(status_line("Итог", "запусков еще не было", WARNING))
    else:
        print(colorize("Последний результат", BOLD + ACCENT_SOFT))
        print_scheduler_run_details(last_run)

    print()
    print(colorize("Следующие задачи", BOLD + ACCENT_SOFT))
    if not next_runs:
        print(status_line("Очередь", "нет активных задач", WARNING))
        return

    for job in next_runs[:5]:
        if isinstance(job, dict):
            print(status_line(str(job.get("title", "")), str(job.get("next_run_at", "")), WARNING))


def print_scheduler_run_line(run: dict[str, Any]) -> None:
    status = str(run.get("status", ""))
    status_color = SUCCESS if status == "success" else ERROR
    title = str(run.get("job_title", ""))
    print(status_line(title or "job", status, status_color))
    message = scheduler_run_message(run)
    if message:
        print(indented_line(message, level=2))


def print_scheduler_run_details(run: dict[str, Any]) -> None:
    status = str(run.get("status", ""))
    status_color = SUCCESS if status == "success" else ERROR
    result = run.get("result")
    result_payload = result if isinstance(result, dict) else {}
    print(status_line("Job", str(run.get("job_title", "")), VALUE))
    print(status_line("Тип", str(run.get("job_kind", "")), VALUE))
    print(status_line("Статус", status, status_color))
    print(status_line("Завершен", str(run.get("finished_at", "")), VALUE))
    if result_payload.get("type") == "llm_summary":
        print(status_line("Источник", "LLM-generated summary", SUCCESS))
        model = result_payload.get("model")
        if model:
            print(status_line("Модель", str(model), VALUE))
        usage = result_payload.get("usage")
        if isinstance(usage, dict) and usage:
            total = usage.get("total_tokens")
            if total is not None:
                print(status_line("Tokens", str(total), VALUE))
    message = scheduler_run_message(run)
    if message:
        print(status_line("Результат", message, SUCCESS if status == "success" else ERROR))
    error = run.get("error")
    if error:
        print(status_line("Ошибка", str(error), ERROR))


def scheduler_run_message(run: dict[str, Any]) -> str:
    result = run.get("result")
    if not isinstance(result, dict):
        return ""
    message = result.get("message")
    if isinstance(message, str):
        return message
    summary = result.get("summary")
    if isinstance(summary, str):
        return summary
    return ""


def count_runs_by_status(runs: list[Any], status: str) -> int:
    return sum(1 for run in runs if isinstance(run, dict) and run.get("status") == status)


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def first_dict(values: list[Any]) -> dict[str, Any] | None:
    for value in values:
        if isinstance(value, dict):
            return value
    return None


def format_seconds(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600} ч"
    if seconds % 60 == 0:
        return f"{seconds // 60} мин"
    return f"{seconds} сек"


def print_mcp_config_status(
    config: MCPConfig,
    timeout: float,
    *,
    show_errors: bool = False,
    show_hints: bool = False,
) -> bool:
    print(header_line("MCP"))
    print(status_line("Конфиг", str(config.path), VALUE))
    print(status_line("Серверов", str(len(config.servers)), VALUE))

    if not config.servers:
        print(status_line("Статус", "серверы не настроены", WARNING))
        return True

    checks = check_mcp_config_servers(config, timeout)
    name_width = max(len(check.server.name) for check in checks)
    success_count = sum(1 for check in checks if check.ok)
    total_tools = sum(len(check.tools) for check in checks)

    print()
    for check in checks:
        state = mcp_connection_label(check.ok)
        tool_count = len(check.tools) if check.ok else 0
        print(indented_line(f"{check.server.name.ljust(name_width)}  {state}  {tool_count} инструментов"))
        if show_errors and check.error:
            print(indented_line(f"Ошибка: {check.error}", level=2))
            if show_hints:
                print_mcp_server_install_hint(check.server, level=2)

    print()
    color = SUCCESS if success_count == len(config.servers) else ERROR
    print(status_line("Connected servers", f"{success_count} / {len(config.servers)}", color))
    print(status_line("Инструментов", str(total_tools), VALUE))
    return success_count == len(config.servers)


def check_mcp_config_servers(config: MCPConfig, timeout: float) -> list[MCPServerCheck]:
    return [check_mcp_server(server, timeout) for server in config.servers]


def check_mcp_server(server: MCPServerConfig, timeout: float) -> MCPServerCheck:
    try:
        tools = asyncio.run(
            list_mcp_tools(
                server.command,
                server.args,
                cwd=server.cwd,
                env=server.env,
                timeout=timeout,
            )
        )
    except FileNotFoundError:
        return MCPServerCheck(server=server, tools=[], error=f"command not found: {server.command}")
    except MCPConnectionError as error:
        return MCPServerCheck(server=server, tools=[], error=str(error))
    except Exception as error:
        return MCPServerCheck(server=server, tools=[], error=str(error))

    return MCPServerCheck(server=server, tools=tools)


def mcp_connection_label(connected: bool) -> str:
    text = "Connected" if connected else "Not Connected"
    color = SUCCESS if connected else ERROR
    return colorize(text, color)


def print_mcp_server_tools(check: MCPServerCheck) -> None:
    print(command_line(f"{check.server.name}:"))
    if check.error:
        print(indented_line(f"Соединение: {mcp_connection_label(False)} - {check.error}"))
        print_mcp_server_install_hint(check.server)
        return

    print(indented_line(f"Соединение: {mcp_connection_label(True)}"))
    print(indented_line(f"Инструментов: {len(check.tools)}"))
    for tool in check.tools:
        title = f" ({tool.title})" if tool.title else ""
        print(indented_line(f"- {tool.name}{title}", level=2))


def print_mcp_server_install_hint(server: MCPServerConfig, *, level: int = 1) -> None:
    if server.name == "apple-mcp":
        print(indented_line("Подсказка: установите Bun и повторите проверку.", level=level))
        print(indented_line("Команда: curl -fsSL https://bun.sh/install | bash", level=level + 1))
    elif server.name == "cupertino":
        print(indented_line("Подсказка: установите Cupertino и один раз выполните setup.", level=level))
        print(indented_line("Команда: bash <(curl -sSL https://raw.githubusercontent.com/mihaelamj/cupertino/main/install.sh)", level=level + 1))
        print(indented_line("Затем: cupertino setup", level=level + 1))


def build_mcp_server_command(server_path: Path) -> tuple[str, list[str]]:
    normalized_path = str(server_path)
    suffix = server_path.suffix.lower()
    if suffix == ".py":
        return sys.executable, [normalized_path]
    if suffix == ".js":
        return "node", [normalized_path]
    raise ValueError("MCP server script должен быть .py или .js.")


def print_mcp_tools(tools: list[MCPTool]) -> None:
    print(header_line("MCP"))
    print(status_line("Соединение", "Connected", SUCCESS))
    print(status_line("Инструментов", str(len(tools)), VALUE))

    if not tools:
        return

    print()
    for index, tool in enumerate(tools, start=1):
        title = f" ({tool.title})" if tool.title else ""
        print(command_line(f"{index}. {tool.name}{title}"))
        if tool.description:
            print(indented_line(f"Описание: {tool.description}"))
        print_mcp_tool_input(tool.input_schema)
        if index < len(tools):
            print()


def print_mcp_tool_input(schema: dict[str, Any]) -> None:
    parameters = readable_schema_parameters(schema)
    if not parameters:
        print(indented_line("Вход: нет параметров"))
        return

    print(indented_line("Вход:"))
    for parameter in parameters:
        print(indented_line(f"- {parameter}", level=2))


def readable_schema_parameters(schema: dict[str, Any]) -> list[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return []

    required_values = schema.get("required", [])
    required = set(required_values if isinstance(required_values, list) else [])
    parameters: list[str] = []

    for name, raw_definition in properties.items():
        if not isinstance(name, str):
            continue
        definition = raw_definition if isinstance(raw_definition, dict) else {}
        type_label = schema_type_label(definition)
        required_label = "required" if name in required else "optional"
        description = definition.get("description")
        line = f"{name}: {type_label}, {required_label}"
        if isinstance(description, str) and description:
            line = f"{line} - {description}"
        parameters.append(line)

    return parameters


def schema_type_label(definition: dict[str, Any]) -> str:
    value = definition.get("type")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if "anyOf" in definition:
        return " | ".join(
            schema_type_label(item)
            for item in definition["anyOf"]
            if isinstance(item, dict)
        ) or "any"
    if "enum" in definition:
        return "enum"
    return "any"


def indented_line(text: str, level: int = 1) -> str:
    line = f"{'  ' * level}{text}"
    if not use_color():
        return line
    return f"{MUTED}{line}{RESET}"


def run_interactive_session(agent: CodeAgent) -> None:
    print(colorize("Code Agent CLI", BOLD + ACCENT))
    print_startup_summary(agent)

    while True:
        try:
            text = input(prompt()).strip()
            reset_terminal_color()
        except (EOFError, KeyboardInterrupt):
            reset_terminal_color()
            print()
            return

        if not text:
            continue

        if text in {"/exit", "/quit"}:
            return

        if text == "/reset":
            try:
                agent.reset_agent()
            except OSError as error:
                print(f"Ошибка: {error}", file=sys.stderr)
                continue
            print("История, память, профиль и ветки очищены. Инварианты сохранены.")
            continue

        if text == "/help":
            print_help()
            continue

        command, _, argument = text.partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command == "/status":
            print_status(agent)
            continue

        if command in {"/tokens", "/token"}:
            print_current_token_state(agent, argument or None)
            continue

        if command == "/context":
            handle_context_command(agent, argument)
            continue

        if command == "/memory":
            handle_memory_command(agent, argument)
            continue

        if command == "/task":
            handle_task_command(agent, argument)
            continue

        if command == "/profile":
            handle_profile_command(agent, argument)
            continue

        if command in {"/invariants", "/invariant"}:
            handle_invariants_command(agent, argument)
            continue

        if command == "/branch":
            handle_branch_command(agent, argument)
            continue

        if command == "/mcp":
            handle_mcp_command(agent, argument)
            continue

        if warn_bare_scheduler_tool(text):
            continue

        send(agent, text)


def build_prompt(args: argparse.Namespace) -> PromptPayload:
    user_prompt = " ".join(args.prompt).strip()

    if args.file is None:
        return PromptPayload(request_text=user_prompt)

    file_content = read_attached_file(
        args.file,
        args.line_range,
        args.max_file_bytes,
        args.force_file,
    )
    range_note = f", строки {args.line_range}" if args.line_range else ""

    request_text = f"""{user_prompt}

Код из файла {args.file}{range_note}:
```text
{file_content}
```"""
    history_text = f"{user_prompt}\n\n[Файл {args.file}{range_note} был приложен к запросу.]"
    return PromptPayload(request_text=request_text, history_text=history_text)


def send(agent: CodeAgent, prompt: str | PromptPayload) -> bool:
    payload = enrich_prompt_with_mcp_tool_context(normalize_prompt(prompt))
    if is_natural_mcp_orchestration_request(payload.request_text):
        print()
        run_mcp_orchestration_from_command(agent, payload.request_text)
        print()
        return True

    pipeline_request = parse_natural_pipeline_request(payload.request_text)
    if pipeline_request is not None:
        print()
        print(status_line("Pipeline intent", "search -> summarize -> save", SUCCESS))
        print(status_line("Query", pipeline_request["query"], VALUE))
        print(status_line("File", pipeline_request["filename"], VALUE))
        print()
        call_pipeline_tool_from_short_command(
            default_mcp_config_file(),
            "run",
            {
                "query": pipeline_request["query"],
                "filename": pipeline_request["filename"],
            },
        )
        print()
        return True

    fast_answer = agent.handle_memory_only_message(
        payload.request_text,
        history_text=payload.history_text,
    )
    if fast_answer is not None:
        print()
        print_agent_answer(fast_answer)
        print()
        return True

    estimated_tokens = agent.estimate_tokens(
        payload.request_text,
        history_text=payload.history_text,
    )
    if not estimated_tokens.fits_context:
        print_context_limit_error(estimated_tokens)
        return False

    try:
        with loader("Думаю"):
            answer = agent.send_message(
                payload.request_text,
                history_text=payload.history_text,
            )
    except ContextLimitExceededError as error:
        print_context_limit_error(error.breakdown)
        return False
    except MissingAPIKeyError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return False
    except APIRequestError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return False
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return False

    print()
    print_agent_answer(answer)
    print()
    print_last_token_report(agent)
    print()
    return True


def is_natural_mcp_orchestration_request(text: str) -> bool:
    lowered = " ".join(text.strip().lower().split())
    if not lowered:
        return False
    has_search = any(marker in lowered for marker in ("найди", "найти", "поищи", "search", "find"))
    has_save = any(marker in lowered for marker in ("сохрани", "сохранить", "запиши", "save", "заметки"))
    has_schedule = any(marker in lowered for marker in ("напомин", "remind", "проверить завтра", "завтра"))
    has_mcp_scope = any(marker in lowered for marker in ("apple", "cupertino", "swiftui", "ios", "mcp"))
    return has_search and has_save and has_schedule and has_mcp_scope


def normalize_prompt(prompt: str | PromptPayload) -> PromptPayload:
    if isinstance(prompt, PromptPayload):
        return prompt
    return PromptPayload(request_text=prompt)


def parse_natural_pipeline_request(text: str) -> dict[str, str] | None:
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    if not normalized:
        return None

    has_search_intent = any(
        marker in lowered
        for marker in (
            "найди",
            "найти",
            "поищи",
            "search",
            "find",
        )
    )
    has_save_intent = any(
        marker in lowered
        for marker in (
            "сохрани",
            "сохранить",
            "запиши",
            "save",
        )
    )
    if not has_search_intent or not has_save_intent:
        return None

    query = normalized
    query = re.sub(r"^(найди|найти|поищи)\s+(мне\s+)?", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^(search|find)\s+(for\s+)?", "", query, flags=re.IGNORECASE)
    query = re.split(
        r"\s+(?:и\s+)?(?:сохрани|сохранить|запиши|save)\b",
        query,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,:;-")
    if not query:
        return None

    filename = extract_pipeline_filename(normalized, query)
    return {
        "query": query,
        "filename": filename,
    }


def extract_pipeline_filename(text: str, query: str) -> str:
    file_match = re.search(
        r"(?:в\s+файл|файл|to\s+file)\s+([A-Za-z0-9._-]+\.(?:md|txt))",
        text,
        flags=re.IGNORECASE,
    )
    if file_match:
        return file_match.group(1)

    notes_match = re.search(r"\b(?:в\s+заметки|заметки|notes)\b", text, flags=re.IGNORECASE)
    if notes_match:
        return "notes.md"

    slug = re.sub(r"[^A-Za-zА-Яа-я0-9]+", "-", query.lower()).strip("-")
    if not slug:
        slug = "pipeline-result"
    return f"{slug[:48]}.md"


def enrich_prompt_with_mcp_tool_context(payload: PromptPayload) -> PromptPayload:
    mock_user_id = extract_mock_user_id(payload.request_text)
    if mock_user_id is None:
        return payload

    try:
        config = load_mcp_config(default_mcp_config_file())
    except MCPConfigError:
        return payload

    server = find_mcp_server(config, "mock-api")
    if server is None:
        return payload

    arguments = {"user_id": mock_user_id}
    try:
        result = asyncio.run(
            call_mcp_tool(
                server.command,
                server.args,
                "get_mock_user",
                arguments,
                cwd=server.cwd,
                env=server.env,
                timeout=env_float("CODE_AGENT_MCP_TIMEOUT", 30.0),
            )
        )
    except Exception as error:
        tool_context = f"MCP tool mock-api/get_mock_user failed: {error}"
    else:
        tool_context = result.as_text()

    request_text = f"""{payload.request_text}

Контекст MCP-инструмента:
Server: mock-api
Tool: get_mock_user
Arguments: {json.dumps(arguments, ensure_ascii=False)}
Result:
```json
{tool_context}
```

Ответь на исходный запрос пользователя, используя результат MCP-инструмента. Не выдумывай поля, которых нет в результате."""

    history_note = (
        payload.history_text or payload.request_text
    ) + "\n\n[MCP tool mock-api/get_mock_user был вызван и использован в ответе.]"
    return PromptPayload(request_text=request_text, history_text=history_note)


def extract_mock_user_id(text: str) -> int | None:
    patterns = (
        r"\bmock\s+user\s*#?(\d+)\b",
        r"\bjsonplaceholder\s+user\s*#?(\d+)\b",
        r"\bmock[-\s]*(?:api\s+)?(?:пользователь|юзер)\s*#?(\d+)\b",
        r"\b(?:пользователь|юзер)\s+из\s+mock\s+api\s*#?(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


@contextmanager
def loader(label: str) -> Iterator[None]:
    if not sys.stdout.isatty():
        yield
        return

    done = threading.Event()

    def animate() -> None:
        for frame in itertools.cycle((".", "..", "...")):
            if done.is_set():
                break
            print(f"\r{label}{frame}   ", end="", flush=True)
            time.sleep(0.35)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()

    try:
        yield
    finally:
        done.set()
        thread.join()
        print("\r" + " " * (len(label) + 6) + "\r", end="", flush=True)


def print_help() -> None:
    print_section(
        "Использование",
        (
            "agent",
            'agent "объясни, чем struct отличается от class в Swift"',
            'agent --file Sources/App.swift "найди ошибки"',
            'agent --file Sources/App.swift --range 40:120 "проверь этот участок"',
            "agent --mcp-tools path/to/server.py",
        ),
    )
    print()
    print_section(
        "Переменная окружения",
        ('export DEEPSEEK_API_KEY="ваш_ключ"',),
    )
    print()
    print_command_help_section(
        "Базовые команды",
        (
            ("/help", "показать помощь"),
            ("/status", "показать настройки текущей сессии"),
            ("/reset", "очистить историю, память, профиль и ветки; инварианты сохраняются"),
            ("/exit", "выйти"),
        )
    )
    print()
    print_command_help_section(
        "Контекст и токены",
        (
            ("/tokens", "показать токены истории и последнего запроса"),
            ("/tokens текст", "посчитать токены текста без отправки в API"),
            ("/context", "показать стратегию контекста и memory layers"),
            ("/context strategy NAME", "переключить: sliding, memory, branching"),
        )
    )
    print()
    print_command_help_section(
        "Состояние задачи",
        (
            ("/task", "показать формальное состояние задачи"),
            ("/task set stage NAME", "перейти к этапу: planning, execution, validation, done, paused"),
            ("/task set step TEXT", "установить текущий шаг"),
            ("/task set expected TEXT", "установить ожидаемое действие"),
            ("/task set summary TEXT", "установить краткое описание задачи"),
            ("/task pause", "поставить задачу на паузу"),
            ("/task resume", "продолжить задачу с прошлого этапа"),
            ("/task clear", "очистить формальное состояние задачи"),
        )
    )
    print()
    print_command_help_section(
        "Память",
        (
            ("/memory", "показать short-term, working и long-term память"),
            ("/memory short|working|long", "показать отдельный слой памяти"),
            ("/memory clear short", "очистить краткосрочную память диалога"),
            ("/memory clear working", "очистить рабочую память текущей задачи"),
            ("/memory clear long", "очистить долговременную память"),
            ("/memory clear all", "очистить всю память: short-term, working и long-term"),
        )
    )
    print()
    print_command_help_section(
        "Инварианты",
        (
            ("/invariants", "показать обязательные ограничения ассистента"),
            ("/invariants add TEXT", "добавить инвариант"),
            ("/invariants delete N", "удалить инвариант по номеру"),
            ("/invariants clear", "очистить список инвариантов"),
            ("/invariants path", "показать путь к invariants.md"),
        )
    )
    print()
    print_command_help_section(
        "Профиль",
        (
            ("/profile", "показать профиль пользователя из profile.md"),
            ("/profile path", "показать путь к profile.md"),
            ("/profile clear", "очистить profile.md"),
        )
    )
    print()
    print_command_help_section(
        "Ветки",
        (
            ("/branch list", "показать ветки"),
            ("/branch compare A B", "сравнить две ветки"),
            ("/branch checkpoint NAME", "сохранить checkpoint активной ветки"),
            ("/branch create NAME [CHECKPOINT]", "создать ветку"),
            ("/branch switch NAME", "переключиться на ветку"),
        )
    )
    print()
    print_command_help_grouped_section(
        "MCP",
        (
            (
                "Config",
                (
                    ("/mcp", "проверить подключение MCP-серверов из config"),
                    ("/mcp show", "показать сохраненные MCP-серверы"),
                    ("/mcp tools", "показать инструменты MCP-серверов"),
                    ("/mcp test", "проверить MCP-серверы с диагностикой ошибок"),
                    ("/mcp path", "показать путь к MCP config"),
                    ("/mcp add NAME -- COMMAND ARGS", "добавить свой MCP-сервер"),
                    ("/mcp remove NAME", "удалить MCP-сервер из config"),
                    ("/mcp clear", "удалить все MCP-серверы из config"),
                    ("/mcp call SERVER TOOL JSON", "вызвать MCP-инструмент напрямую"),
                    ("/mcp help", "показать помощь по MCP"),
                    ("agent --mcp-config-tools", "проверить MCP config из shell"),
                ),
            ),
            (
                "Init",
                (
                    ("/mcp init-apple", "создать config для apple-mcp и cupertino"),
                    ("/mcp init-mock", "подключить встроенный mock HTTP API MCP-сервер"),
                    ("/mcp init-scheduler", "подключить встроенный SQLite MCP-планировщик"),
                    ("/mcp init-pipeline", "подключить web+LLM MCP pipeline"),
                    ("/mcp init-orchestration", "подключить apple-mcp, cupertino, pipeline и scheduler"),
                ),
            ),
            (
                "Scheduler",
                (
                    ("/mcp remind TEXT AT", "создать reminder без JSON"),
                    ("/mcp run_due", "выполнить due jobs scheduler"),
                    ("/mcp summary", "показать сводку scheduler"),
                    ("/mcp clear-scheduler", "очистить jobs и историю scheduler"),
                ),
            ),
            (
                "Pipeline и RAG",
                (
                    ("/mcp pipeline QUERY FILE", "запустить search -> summarize -> save"),
                    ("/mcp index-docs PATH", "построить локальный индекс документов через Ollama embeddings"),
                    ("/mcp index-status", "показать статус локального индекса документов"),
                    ("/mcp compare-chunking", "сравнить fixed и structural chunking"),
                    ("/mcp rag-search QUESTION", "enhanced search: query rewrite, similarity filter и heuristic rerank"),
                    ("/mcp rag-answer QUESTION", "ответить с verified sources/quotes или сказать Не знаю"),
                    ("/mcp rag-compare QUESTION", "сравнить Without RAG, Baseline RAG и Enhanced RAG"),
                    ("/mcp rag-eval", "проверить sources, quotes и answer/quote alignment на 10 вопросах"),
                    ("/mcp orchestrate TEXT", "построить и выполнить multi-server MCP flow"),
                ),
            ),
        ),
    )
    print()
    print_ollama_help()


def print_ollama_help() -> None:
    print_command_help_section(
        "Ollama",
        (
            ("Модель", "nomic-embed-text для embeddings в /mcp index-docs"),
            ("ollama serve", "запустить локальный server на 127.0.0.1:11434"),
            ("Ctrl+C", "остановить server, если он запущен вручную через ollama serve"),
            ("address already in use", "Ollama уже запущена; второй server на том же порту не стартует"),
        ),
    )


def print_section(title: str, lines: tuple[str, ...]) -> None:
    print(header_line(title))
    for line in lines:
        print(command_line(line))


def print_command_help_section(title: str, commands: tuple[tuple[str, str], ...]) -> None:
    print(header_line(title))
    print_command_help_rows(commands)


def print_command_help_grouped_section(
    title: str,
    groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
) -> None:
    print(header_line(title))
    for index, (group_title, commands) in enumerate(groups):
        if index:
            print()
        print(subheader_line(group_title))
        print_command_help_rows(commands, indent=4)


def print_command_help_rows(commands: tuple[tuple[str, str], ...], *, indent: int = 2) -> None:
    width = max(len(command) for command, _ in commands)
    description_width = max(DEFAULT_WRAP_WIDTH - indent - width - 2, 32)
    for command, description in commands:
        wrapped_description = textwrap.wrap(description, width=description_width) or [""]
        if not use_color():
            print(f"{' ' * indent}{command:<{width}}  {wrapped_description[0]}")
            for line in wrapped_description[1:]:
                print(f"{' ' * indent}{'':<{width}}  {line}")
            continue
        print(f"{' ' * indent}{COMMAND}{command:<{width}}{RESET}  {MUTED}{wrapped_description[0]}{RESET}")
        for line in wrapped_description[1:]:
            print(f"{' ' * indent}{'':<{width}}  {MUTED}{line}{RESET}")


def print_help_examples(examples: tuple[str, ...], *, indent: int = 4) -> None:
    width = max(DEFAULT_WRAP_WIDTH - indent, 40)
    for example in examples:
        wrapped_lines = textwrap.wrap(example, width=width, subsequent_indent="  ") or [example]
        for line in wrapped_lines:
            print(indented_line(line, level=indent // 2))


def header_line(text: str) -> str:
    if not use_color():
        return f"{text}:"
    return f"{SUCCESS}{BOLD}{text}{RESET}{SUBTLE}:{RESET}"


def subheader_line(text: str) -> str:
    if not use_color():
        return f"  {text}:"
    return f"  {CODE_KEYWORD}{BOLD}{text}{RESET}{SUBTLE}:{RESET}"


def command_line(text: str) -> str:
    if not use_color():
        return f"  {text}"
    return f"  {COMMAND}{text}{RESET}"


def print_status(agent: CodeAgent) -> None:
    status = agent.status()
    api_key_status = "задан" if status["api_key_configured"] else "не задан"
    history = f"{status['history_messages']} / {status['max_history_messages']}"
    history_loaded = "загружена" if status["history_loaded"] else "новая"

    print(header_line("Модель"))
    for line in (
        status_line("Модель", str(status["model"])),
        status_line("Temperature", str(status["temperature"])),
        status_line("API URL", str(status["api_url"])),
        status_line(
            "DEEPSEEK_API_KEY",
            api_key_status,
            SUCCESS if status["api_key_configured"] else WARNING,
        ),
    ):
        print(line)

    print()
    print(header_line("Контекст"))
    for line in (
        status_line("Лимит контекста", f"{status['context_limit']} токенов"),
        status_line("Токены истории", str(status["current_history_tokens"])),
        status_line("Остаток контекста", str(status["remaining_context_tokens"])),
    ):
        print(line)

    print()
    print(header_line("Стратегия"))
    for line in (
        status_line("Режим", str(status["context_strategy"])),
        status_line("Активная ветка", str(status["active_branch"])),
        status_line("Веток", str(status["branch_count"])),
        status_line("Working memory", f"{status['working_memory_count']} ключей"),
        status_line("Long-term memory", f"{status['long_term_memory_count']} ключей"),
        status_line("Invariants", f"{status['invariant_count']} правил"),
        status_line("Task stage", str(status["task_stage"])),
        status_line("Memory tokens", str(status["memory_tokens"])),
        status_line("Memory max", f"{status['memory_max_tokens']} токенов"),
        status_line("Auto memory", "on" if status["auto_memory_updates"] else "off"),
        status_line("Auto task state", "on" if status["auto_task_state_updates"] else "off"),
    ):
        print(line)
    if status["task_current_step"]:
        print(status_line("Task step", str(status["task_current_step"])))
    if status["task_expected_action"]:
        print(status_line("Task expected", str(status["task_expected_action"])))

    if status["last_memory_error"]:
        print(status_line("Ошибка memory", str(status["last_memory_error"]), WARNING))
    if status["last_invariant_error"]:
        print(status_line("Ошибка invariant", str(status["last_invariant_error"]), WARNING))
    if status["last_task_transition_error"]:
        print(status_line("Ошибка task transition", str(status["last_task_transition_error"]), WARNING))

    print()
    print(header_line("История"))
    for line in (
        status_line("Сообщения", f"{history} сообщений"),
        status_line("Файл истории", str(status["history_file"])),
        status_line("Файл профиля", str(status["profile_file"])),
        status_line("Файл инвариантов", str(status["invariants_file"])),
        status_line("Состояние истории", history_loaded),
    ):
        print(line)

    print()
    print(header_line("Сессия"))
    for line in (
        status_line("Токены сессии", str(status["session_total_tokens"]), VALUE),
        status_line("Prompt", str(status["session_prompt_tokens"])),
        status_line("Answer", str(status["session_completion_tokens"])),
    ):
        print(line)

    if agent.last_token_breakdown is None:
        print(status_line("Последний запрос", "ещё не отправлялся", WARNING))
    else:
        print()
        print_last_token_report(agent)


def status_line(label: str, value: str, value_color: str = VALUE) -> str:
    if not use_color():
        return f"{label}: {value}"
    return f"{MUTED}{label}{SUBTLE}:{RESET} {value_color}{value}{RESET}"


def print_startup_summary(agent: CodeAgent) -> None:
    status = agent.status()
    history_state = "загружена" if status["history_loaded"] else "новая"

    print(status_line("Модель", str(status["model"])))
    print(status_line("Стратегия", str(status["context_strategy"])))
    print(status_line("Ветка", str(status["active_branch"])))
    print(status_line("История", f"{status['history_messages']}/{status['max_history_messages']} · {history_state}"))
    print(status_line("Контекст", f"{status['current_history_tokens']}/{status['context_limit']} токенов"))
    print(status_line("MCP", mcp_startup_status()))
    print(colorize("Введите /help для списка команд.", MUTED))


def mcp_startup_status() -> str:
    try:
        config = load_mcp_config(default_mcp_config_file())
    except MCPConfigError:
        config_path = default_mcp_config_file()
        if config_path.exists():
            return "ошибка config"
        return "не настроен"
    if not config.servers:
        return "не настроен"
    return f"настроено серверов: {len(config.servers)}"


def print_current_token_state(agent: CodeAgent, request_text: str | None = None) -> None:
    status = agent.status()
    print(header_line("Текущие токены"))
    for line in (
        status_line("Вся история диалога", str(status["current_history_tokens"])),
        status_line("Лимит модели", str(status["context_limit"])),
        status_line("Остаток", str(status["remaining_context_tokens"])),
        status_line("Стратегия", str(status["context_strategy"])),
        status_line("Активная ветка", str(status["active_branch"])),
        status_line("Memory tokens", str(status["memory_tokens"])),
        status_line("Invariant tokens", str(agent.context_report()["invariant_tokens"])),
        status_line("Сессия total", str(status["session_total_tokens"]), VALUE),
        status_line("Сессия prompt", str(status["session_prompt_tokens"])),
        status_line("Сессия answer", str(status["session_completion_tokens"])),
    ):
        print(line)

    if request_text:
        print()
        print_token_estimate(agent.estimate_tokens(request_text))
        return

    if agent.last_token_breakdown is not None:
        print()
        print_last_token_report(agent)
    else:
        print(status_line("Последний запрос", "ещё не отправлялся", WARNING))


def print_last_token_report(agent: CodeAgent) -> None:
    breakdown = agent.last_token_breakdown
    if breakdown is None:
        return

    usage = agent.last_actual_usage
    actual_prompt = int(usage.get("prompt_tokens") or 0)
    actual_answer = int(usage.get("completion_tokens") or 0)
    actual_total = int(usage.get("total_tokens") or 0)
    full_history_tokens = agent.token_counter.count_messages(agent.messages)

    print(header_line("Токены"))
    print(status_line("Текущий запрос", f"{breakdown.current_request_tokens} (локальная оценка)"))
    print(status_line("Вся история диалога", f"{full_history_tokens} (локальная оценка)"))
    report = agent.context_report()
    if report["memory_tokens"]:
        print(status_line("Memory layers", f"{report['memory_tokens']} токенов"))
    if report["invariant_tokens"]:
        print(status_line("Invariants", f"{report['invariant_tokens']} токенов"))
    if actual_total:
        print(status_line("Ответ модели", f"{actual_answer} (API)", SUCCESS))
        print()
        print(header_line("Детали API"))
        print(status_line("Prompt целиком", str(actual_prompt), VALUE))
        print(status_line("Total", str(actual_total), VALUE))
        prompt_cost = actual_prompt * agent.input_price_per_1m / 1_000_000
        answer_cost = actual_answer * agent.output_price_per_1m / 1_000_000
        print(status_line("Стоимость prompt API", format_usd(prompt_cost), MONEY))
        print(status_line("Стоимость answer", format_usd(answer_cost), MONEY))
        print(status_line("Стоимость total", format_usd(prompt_cost + answer_cost), MONEY + BOLD))
    else:
        print(status_line("Ответ модели", "нет данных usage от API", WARNING))
        print()
        print(header_line("Детали оценки"))
        print(status_line("Prompt целиком", str(breakdown.prompt_tokens)))
        print(status_line("Стоимость prompt оценка", format_usd(breakdown.input_cost_usd)))
    if not breakdown.fits_context:
        print(status_line("Переполнение", f"+{breakdown.overflow_tokens} токенов", WARNING))


def print_token_estimate(breakdown: TokenBreakdown) -> None:
    print(header_line("Оценка токенов"))
    for line in (
        status_line("Текущий запрос", str(breakdown.current_request_tokens)),
        status_line("История до запроса", str(breakdown.history_tokens)),
        status_line("Prompt к модели", str(breakdown.prompt_tokens), VALUE),
        status_line("Лимит модели", str(breakdown.context_limit)),
        status_line("Остаток контекста", str(breakdown.remaining_context_tokens)),
        status_line("Стоимость prompt", format_usd(breakdown.input_cost_usd), MONEY),
    ):
        print(line)

    if not breakdown.fits_context:
        print(status_line("Переполнение", f"+{breakdown.overflow_tokens} токенов", WARNING))


def print_context_limit_error(breakdown: TokenBreakdown) -> None:
    print(colorize("Запрос не отправлен: превышен лимит контекстного окна.", WARNING))
    print()
    print(header_line("Токены"))
    for line in (
        status_line("Текущий запрос", str(breakdown.current_request_tokens)),
        status_line("История до запроса", str(breakdown.history_tokens)),
        status_line("Prompt к модели", str(breakdown.prompt_tokens), VALUE),
        status_line("Лимит модели", str(breakdown.context_limit)),
        status_line("Превышение", f"+{breakdown.overflow_tokens} токенов", WARNING),
    ):
        print(line)
    print()
    print(header_line("Что можно сделать"))
    for line in (
        "/reset — очистить историю диалога",
        "сократить текущий запрос или приложенный файл",
        "использовать --range для большого файла",
        "увеличить CODE_AGENT_CONTEXT_LIMIT, если модель реально поддерживает больший контекст",
    ):
        print(command_line(line))


def handle_context_command(agent: CodeAgent, argument: str) -> None:
    parts = argument.split()
    if not parts:
        print_context_report(agent)
        return

    if len(parts) == 2 and parts[0] == "strategy":
        agent.set_context_strategy(parts[1])
        print(status_line("Стратегия", agent.context_strategy, SUCCESS))
        return

    print("Использование: /context или /context strategy sliding|memory|branching")


def print_context_report(agent: CodeAgent) -> None:
    report = agent.context_report()
    print(header_line("Контекст"))
    for line in (
        status_line("Стратегия", str(report["strategy"])),
        status_line("Активная ветка", str(report["active_branch"])),
        status_line("Сообщения", f"{report['messages']} / {report['max_messages']}"),
        status_line("Prompt tokens", str(report["prompt_tokens_current_strategy"])),
        status_line("Sliding tokens", str(report["prompt_tokens_sliding"])),
        status_line("Memory tokens", str(report["memory_tokens"])),
        status_line("Invariant tokens", str(report["invariant_tokens"])),
    ):
        print(line)

    if report["last_memory_error"]:
        print(status_line("Ошибка memory", str(report["last_memory_error"]), WARNING))
    if report["last_invariant_error"]:
        print(status_line("Ошибка invariant", str(report["last_invariant_error"]), WARNING))
    if report["last_task_transition_error"]:
        print(status_line("Ошибка task transition", str(report["last_task_transition_error"]), WARNING))

    print()
    print_task_state(agent)
    print()
    print_memory_layers(agent, include_short=False)
    print()
    print_invariants(agent)


def handle_task_command(agent: CodeAgent, argument: str) -> None:
    parts = argument.split()
    if not parts:
        print_task_state(agent)
        return

    action = parts[0].lower()
    try:
        if action == "pause" and len(parts) == 1:
            agent.pause_task()
            print(status_line("Task stage", agent.memory.task_state.stage, WARNING))
            return
        if action == "resume" and len(parts) == 1:
            agent.resume_task()
            print(status_line("Task stage", agent.memory.task_state.stage, SUCCESS))
            return
        if action == "clear" and len(parts) == 1:
            agent.clear_task_state()
            print(status_line("Task state", "очищено", WARNING))
            return
        if action == "set" and len(parts) >= 3:
            field_name = parts[1].lower()
            value = " ".join(parts[2:]).strip()
            if field_name == "stage":
                agent.set_task_stage(value)
                print(status_line("Task stage", agent.memory.task_state.stage, SUCCESS))
                return
            if field_name == "step":
                agent.set_task_current_step(value)
                print(status_line("Task step", value, SUCCESS))
                return
            if field_name == "expected":
                agent.set_task_expected_action(value)
                print(status_line("Task expected", value, SUCCESS))
                return
            if field_name == "summary":
                agent.set_task_summary(value)
                print(status_line("Task summary", value, SUCCESS))
                return
            print("Использование: /task set stage|step|expected|summary ...")
            return
    except (CodeAgentError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return

    print(
        "Использование: /task | /task set stage|step|expected|summary ... | "
        "/task pause | /task resume | /task clear"
    )


def print_task_state(agent: CodeAgent) -> None:
    task_state = agent.task_report()
    print(header_line("Task State"))
    print(status_line("stage", task_state.get("stage", "planning")))
    if task_state.get("current_step"):
        print(status_line("current_step", task_state["current_step"]))
    if task_state.get("expected_action"):
        print(status_line("expected_action", task_state["expected_action"]))
    if task_state.get("summary"):
        print(status_line("summary", task_state["summary"]))
    if task_state.get("previous_stage"):
        print(status_line("previous_stage", task_state["previous_stage"], MUTED))
    if task_state.get("allowed_next_stages"):
        print(status_line("allowed_next_stages", task_state["allowed_next_stages"]))
    if task_state.get("guidance"):
        print(status_line("guidance", task_state["guidance"]))
    if task_state.get("next_action"):
        print(status_line("next_action", task_state["next_action"]))


def handle_memory_command(agent: CodeAgent, argument: str) -> None:
    parts = argument.split()
    if not parts:
        print_memory_layers(agent)
        return

    layer = parts[0].lower()
    if layer in {"short", "short-term", "short_term"} and len(parts) == 1:
        print_short_term_memory(agent)
        return

    if layer in {"working", "work"} and len(parts) == 1:
        print_memory_layer("Working memory", agent.memory.working)
        return

    if layer in {"long", "long-term", "long_term"} and len(parts) == 1:
        print_memory_layer("Long-term memory", agent.memory.long_term)
        return

    if len(parts) == 2 and parts[0].lower() == "clear":
        target = parts[1].lower()
        if target in {"short", "short-term", "short_term"}:
            try:
                agent.clear_short_term_memory()
            except OSError as error:
                print(f"Ошибка: {error}", file=sys.stderr)
                return
            print(status_line("Short-term memory", "очищена", WARNING))
            return
        if target in {"working", "work"}:
            try:
                agent.clear_working_memory()
            except OSError as error:
                print(f"Ошибка: {error}", file=sys.stderr)
                return
            print(status_line("Working memory", "очищена", WARNING))
            return
        if target in {"long", "long-term", "long_term"}:
            try:
                agent.clear_long_term_memory()
            except OSError as error:
                print(f"Ошибка: {error}", file=sys.stderr)
                return
            print(status_line("Long-term memory", "очищена", WARNING))
            return
        if target == "all":
            try:
                agent.clear_all_memory()
            except OSError as error:
                print(f"Ошибка: {error}", file=sys.stderr)
                return
            print(status_line("Memory", "полностью очищена", WARNING))
            return

    print(
        "Использование: /memory | /memory short|working|long | "
        "/memory clear short|working|long|all"
    )


def print_memory_layers(agent: CodeAgent, include_short: bool = True) -> None:
    if include_short:
        print_short_term_memory(agent)
        print()
    print_memory_layer("Working memory", agent.memory.working)
    print()
    print_memory_layer("Long-term memory", agent.memory.long_term)


def print_short_term_memory(agent: CodeAgent) -> None:
    visible_messages = [
        message
        for message in agent.messages
        if message.get("role") in {"user", "assistant"}
    ]
    print(header_line("Short-term memory"))
    print(status_line("Сообщения", f"{len(visible_messages)} / {agent.max_history_messages}"))
    if not visible_messages:
        print(status_line("Состояние", "пусто", WARNING))
        return
    for message in visible_messages[-min(len(visible_messages), 6):]:
        role = message.get("role", "")
        content = shorten_text(message.get("content", ""), 96)
        print(status_line(role, content, MUTED))


def print_memory_layer(title: str, layer: dict[str, str]) -> None:
    print(header_line(title))
    if not layer:
        print(status_line("Состояние", "пусто", WARNING))
        return
    for key in sorted(layer):
        print(status_line(key, layer[key]))


def handle_profile_command(agent: CodeAgent, argument: str) -> None:
    parts = argument.split()
    if not parts:
        print_profile(agent)
        return

    action = parts[0].lower()
    if action == "path" and len(parts) == 1:
        print(status_line("Файл профиля", str(agent.profile_storage.path), VALUE))
        return

    if action == "clear" and len(parts) == 1:
        try:
            agent.clear_long_term_memory()
        except OSError as error:
            print(f"Ошибка: {error}", file=sys.stderr)
            return
        print(status_line("Профиль", "очищен", WARNING))
        return

    print(
        "Использование: /profile | /profile path | /profile clear"
    )


def print_profile(agent: CodeAgent) -> None:
    print(header_line("Профиль"))
    print(status_line("Файл", str(agent.profile_storage.path), VALUE))
    print()
    print_memory_layer("Long-term memory", agent.memory.long_term)


def handle_invariants_command(agent: CodeAgent, argument: str) -> None:
    parts = argument.split()
    if not parts:
        print_invariants(agent)
        return

    action = parts[0].lower()
    try:
        if action == "path" and len(parts) == 1:
            print(status_line("Файл инвариантов", str(agent.invariant_storage.path), VALUE))
            return
        if action == "add" and len(parts) >= 2:
            invariant = " ".join(parts[1:]).strip()
            agent.add_invariant(invariant)
            print(status_line("Инвариант добавлен", invariant, SUCCESS))
            return
        if action in {"delete", "del", "remove", "rm"} and len(parts) == 2:
            try:
                index = int(parts[1])
            except ValueError:
                print("Ошибка: номер инварианта должен быть числом.", file=sys.stderr)
                return
            removed = agent.delete_invariant(index)
            print(status_line("Инвариант удален", removed, WARNING))
            return
        if action == "clear" and len(parts) == 1:
            agent.clear_invariants()
            print(status_line("Инварианты", "очищены", WARNING))
            return
    except (CodeAgentError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return

    print(
        "Использование: /invariants | /invariants add TEXT | "
        "/invariants delete N | /invariants clear | /invariants path"
    )


def print_invariants(agent: CodeAgent) -> None:
    report = agent.invariants_report()
    print(header_line("Инварианты"))
    print(status_line("Файл", str(report["path"]), VALUE))
    print(status_line("Количество", str(report["count"])))
    print(status_line("Токены", str(report["tokens"])))
    invariants = report["invariants"]
    if not invariants:
        print(status_line("Состояние", "пусто", WARNING))
        return
    for index, invariant in enumerate(invariants, start=1):
        print(status_line(str(index), invariant))


def handle_branch_command(agent: CodeAgent, argument: str) -> None:
    parts = argument.split()
    if not parts or parts[0] == "list":
        print_branch_report(agent)
        return

    try:
        if len(parts) == 2 and parts[0] == "checkpoint":
            agent.create_checkpoint(parts[1])
            print(status_line("Checkpoint", parts[1], SUCCESS))
            return
        if parts[0] == "create" and len(parts) in {2, 3}:
            agent.create_branch(parts[1], parts[2] if len(parts) == 3 else None)
            print(status_line("Ветка создана", parts[1], SUCCESS))
            return
        if len(parts) == 2 and parts[0] == "switch":
            agent.switch_branch(parts[1])
            print(status_line("Активная ветка", parts[1], SUCCESS))
            return
        if len(parts) == 3 and parts[0] == "compare":
            print_branch_compare(agent, parts[1], parts[2])
            return
        if len(parts) == 2 and parts[0] == "delete":
            agent.delete_branch(parts[1])
            print(status_line("Ветка удалена", parts[1], WARNING))
            return
    except CodeAgentError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return

    print(
        "Использование: /branch list | /branch compare A B | /branch checkpoint NAME | "
        "/branch create NAME [CHECKPOINT] | /branch switch NAME | /branch delete NAME"
    )


def handle_mcp_command(agent: CodeAgent, argument: str) -> None:
    command = argument.strip().lower()
    config_path = default_mcp_config_file()

    if command in {"help", "--help", "-h"}:
        print_mcp_help()
        return

    short_command, _, short_argument = argument.strip().partition(" ")
    if handle_pipeline_short_command(config_path, short_command.lower(), short_argument.strip()):
        return

    if handle_scheduler_short_command(config_path, short_command.lower(), short_argument.strip()):
        return

    if short_command.lower() in {"orchestrate", "orchestration"}:
        run_mcp_orchestration_from_command(agent, short_argument)
        return

    if command.startswith("add "):
        add_mcp_server_from_command(config_path, argument.strip())
        return

    if command.startswith("remove "):
        remove_mcp_server_from_command(config_path, argument.strip())
        return

    if command in {"clear", "disable", "off"}:
        clear_mcp_servers_from_command(config_path)
        return

    if command.startswith("call "):
        call_mcp_tool_from_command(config_path, argument.strip())
        return

    if command in {"show", "config", "list-servers"}:
        try:
            config = load_mcp_config_or_empty(config_path)
        except MCPConfigError as error:
            print_mcp_config_missing(config_path, error if config_path.exists() else None)
            return
        print_mcp_config_servers(config)
        return

    if command in {"", "status"}:
        try:
            config = load_mcp_config(config_path)
        except MCPConfigError as error:
            print_mcp_config_missing(config_path, error if config_path.exists() else None)
            return
        print_mcp_config_status(config, env_float("CODE_AGENT_MCP_TIMEOUT", 30.0))
        return

    if command in {"tools", "list"}:
        try:
            config = load_mcp_config(config_path)
        except MCPConfigError as error:
            print_mcp_config_missing(config_path, error if config_path.exists() else None)
            return
        print_mcp_config_tools(config, env_float("CODE_AGENT_MCP_TIMEOUT", 30.0))
        return

    if command in {"test", "check"}:
        try:
            config = load_mcp_config(config_path)
        except MCPConfigError as error:
            print_mcp_config_missing(config_path, error if config_path.exists() else None)
            return
        print_mcp_config_status(
            config,
            env_float("CODE_AGENT_MCP_TIMEOUT", 30.0),
            show_errors=True,
            show_hints=True,
        )
        return

    if command == "path":
        print(header_line("MCP"))
        print(status_line("Конфиг", str(config_path), VALUE))
        return

    if command in {"init-apple", "init-apple force"}:
        init_apple_mcp_config(config_path, force=command.endswith(" force"))
        return

    if command == "init-mock":
        init_mock_mcp_config(config_path)
        return

    if command == "init-scheduler":
        init_scheduler_mcp_config(config_path)
        return

    if command == "init-pipeline":
        init_pipeline_mcp_config(config_path)
        return

    if command == "init-orchestration":
        init_orchestration_mcp_config(config_path)
        return

    print("Использование: /mcp | /mcp add NAME -- COMMAND ARGS | /mcp remove NAME | /mcp clear | /mcp tools | /mcp remind TEXT AT | /mcp run_due | /mcp summary | /mcp clear-scheduler | /mcp pipeline QUERY FILE | /mcp index-docs PATH | /mcp index-status | /mcp compare-chunking | /mcp rag-search QUESTION | /mcp rag-answer QUESTION | /mcp rag-compare QUESTION | /mcp rag-eval | /mcp orchestrate TEXT | /mcp call SERVER TOOL JSON | /mcp init-mock | /mcp init-scheduler | /mcp init-pipeline | /mcp init-orchestration | /mcp show | /mcp test | /mcp help")


def print_branch_report(agent: CodeAgent) -> None:
    report = agent.branch_report()
    print(header_line("Ветки"))
    print(status_line("Активная", str(report["active_branch"]), SUCCESS))
    for name, branch in report["branches"].items():
        marker = "*" if name == report["active_branch"] else " "
        value = (
            f"{branch['messages']} сообщений, "
            f"{len(branch['working_memory'])} working, "
            f"{len(branch['long_term_memory'])} long-term, "
            f"{branch['task_state'].get('stage', 'planning')} task, "
            f"{branch['prompt_tokens']} prompt tokens, "
            f"{len(branch['checkpoints'])} checkpoints"
        )
        print(status_line(f"{marker} {name}", value))
        if branch["last_user"]:
            print(status_line("  последний user", shorten_text(branch["last_user"], 88), MUTED))
        if branch["current_task"]:
            print(status_line("  current_task", shorten_text(branch["current_task"], 88), MUTED))
        if branch["task_state"].get("current_step"):
            print(status_line("  task_step", shorten_text(branch["task_state"]["current_step"], 88), MUTED))
        elif branch["goal"]:
            print(status_line("  goal", shorten_text(branch["goal"], 88), MUTED))


def print_branch_compare(agent: CodeAgent, left_name: str, right_name: str) -> None:
    report = agent.compare_branches(left_name, right_name)
    left = report["left"]
    right = report["right"]

    print(header_line("Сравнение веток"))
    print(status_line("Левая", str(report["left_name"]), SUCCESS))
    print(status_line("Правая", str(report["right_name"]), SUCCESS))
    print()
    print(header_line("Размер контекста"))
    print(status_line(str(report["left_name"]), f"{left['prompt_tokens']} prompt tokens"))
    print(status_line(str(report["right_name"]), f"{right['prompt_tokens']} prompt tokens"))
    print()
    print(header_line("Последний user"))
    print(status_line(str(report["left_name"]), shorten_text(left["last_user"], 120)))
    print(status_line(str(report["right_name"]), shorten_text(right["last_user"], 120)))

    if left["current_task"] or right["current_task"] or left["goal"] or right["goal"]:
        print()
        print(header_line("Смысл ветки"))
        print(status_line(str(report["left_name"]), shorten_text(left["current_task"] or left["goal"], 120)))
        print(status_line(str(report["right_name"]), shorten_text(right["current_task"] or right["goal"], 120)))

    memory_diff = report["memory_diff"]
    if memory_diff:
        print()
        print(header_line("Разная память"))
        for key, diff in memory_diff.items():
            print(status_line(f"{key} / {report['left_name']}", shorten_text(diff["left"], 110), MUTED))
            print(status_line(f"{key} / {report['right_name']}", shorten_text(diff["right"], 110), MUTED))
    else:
        print()
        print(status_line("Memory", "одинаковая", WARNING))


def format_usd(value: float) -> str:
    if value < 0.0001:
        return f"${value:.6f}"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def read_attached_file(
    file_path: Path,
    line_range: str | None,
    max_file_bytes: int,
    force_file: bool,
) -> str:
    if line_range:
        file_range = parse_file_range(line_range)
        return read_file_range(file_path, file_range)

    try:
        file_size = file_path.stat().st_size
    except OSError as error:
        raise SystemExit(f"Ошибка: не удалось прочитать файл {file_path}: {error}") from error

    if file_size > max_file_bytes and not force_file:
        if not confirm_large_file(file_path, file_size, max_file_bytes):
            raise SystemExit("Отменено: файл не отправлен.")

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"Ошибка: файл {file_path} не похож на UTF-8 текст.") from error
    except OSError as error:
        raise SystemExit(f"Ошибка: не удалось прочитать файл {file_path}: {error}") from error


def read_file_range(file_path: Path, file_range: FileRange) -> str:
    selected_lines: list[str] = []

    try:
        with file_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if line_number < file_range.start:
                    continue
                if line_number > file_range.end:
                    break
                selected_lines.append(line.rstrip("\n"))
    except UnicodeDecodeError as error:
        raise SystemExit(f"Ошибка: файл {file_path} не похож на UTF-8 текст.") from error
    except OSError as error:
        raise SystemExit(f"Ошибка: не удалось прочитать файл {file_path}: {error}") from error

    if not selected_lines:
        raise SystemExit(
            f"Ошибка: в файле {file_path} нет строк в диапазоне "
            f"{file_range.start}:{file_range.end}."
        )

    return "\n".join(selected_lines)


def parse_file_range(value: str) -> FileRange:
    match = re.fullmatch(r"(\d+):(\d+)", value.strip())
    if not match:
        raise SystemExit("Ошибка: --range должен быть в формате START:END, например 40:120.")

    start = int(match.group(1))
    end = int(match.group(2))
    if start < 1 or end < start:
        raise SystemExit("Ошибка: --range должен быть положительным диапазоном START:END.")

    return FileRange(start=start, end=end)


def confirm_large_file(file_path: Path, file_size: int, max_file_bytes: int) -> bool:
    size_kb = file_size / 1024
    limit_kb = max_file_bytes / 1024
    message = (
        f"Файл {file_path} большой: {size_kb:.1f} KB "
        f"(лимит {limit_kb:.1f} KB). Отправить целиком? [y/N] "
    )

    if not sys.stdin.isatty():
        print(
            f"Ошибка: файл {file_path} больше лимита. "
            "Используйте --range или --force-file.",
            file=sys.stderr,
        )
        return False

    answer = input(colorize(message, WARNING)).strip().lower()
    reset_terminal_color()
    return answer in {"y", "yes", "д", "да"}


def print_agent_answer(answer: str) -> None:
    for line in render_answer(answer):
        print(line)


def render_answer(answer: str) -> list[str]:
    rendered: list[str] = []
    text_lines: list[str] = []
    code_lines: list[str] = []
    code_language = ""
    in_code = False

    def flush_text() -> None:
        if not text_lines:
            return
        rendered.extend(render_text_block(text_lines))
        text_lines.clear()

    for line in answer.splitlines():
        if line.startswith("```"):
            if in_code:
                flush_text()
                rendered.extend(render_code_block(code_lines, code_language))
                code_lines = []
                code_language = ""
                in_code = False
            else:
                flush_text()
                in_code = True
                code_language = line.removeprefix("```").strip()
            continue

        if in_code:
            code_lines.append(line)
        else:
            text_lines.append(line)

    if in_code:
        rendered.extend(render_code_block(code_lines, code_language))
    else:
        flush_text()

    return trim_blank_edges(rendered)


def render_text_block(lines: list[str]) -> list[str]:
    rendered: list[str] = []
    paragraph: list[str] = []
    previous_blank = False

    def flush_paragraph() -> None:
        if not paragraph:
            return
        rendered.extend(render_paragraph(" ".join(part.strip() for part in paragraph)))
        paragraph.clear()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            if rendered and not previous_blank:
                rendered.append("")
            previous_blank = True
            continue

        if heading_level(stripped):
            flush_paragraph()
            if rendered and not previous_blank:
                rendered.append("")
            rendered.extend(render_heading(stripped))
            previous_blank = False
            continue

        if list_prefix(stripped):
            flush_paragraph()
            rendered.extend(render_list_item(stripped))
            previous_blank = False
            continue

        paragraph.append(stripped)
        previous_blank = False

    flush_paragraph()
    return rendered


def render_paragraph(text: str) -> list[str]:
    width = content_width()
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    return [colorize(line, ANSWER_TEXT) for line in lines]


def render_heading(line: str) -> list[str]:
    level = heading_level(line)
    title = line[level:].strip()
    if not title:
        return []

    color = ANSWER_ACCENT + (BOLD if level <= 2 else "")
    return [colorize(title, color)]


def heading_level(line: str) -> int:
    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else 0


def render_list_item(line: str) -> list[str]:
    width = content_width()
    prefix = list_prefix(line)
    body = line[len(prefix) :].strip()
    continuation_indent = " " * len(prefix)
    wrapped = textwrap.wrap(
        body,
        width=max(width - len(prefix), 24),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]

    lines = [prefix + wrapped[0]]
    lines.extend(continuation_indent + part for part in wrapped[1:])
    return [colorize(line, ANSWER_TEXT) for line in lines]


def trim_blank_edges(lines: list[str]) -> list[str]:
    while lines and not strip_ansi(lines[0]).strip():
        lines.pop(0)
    while lines and not strip_ansi(lines[-1]).strip():
        lines.pop()
    return lines


def render_text_line(line: str) -> list[str]:
    width = terminal_text_width()
    stripped = line.strip()
    indent = text_indent(line)
    bullet_prefix = list_prefix(stripped)

    if bullet_prefix:
        body = stripped[len(bullet_prefix) :].strip()
        wrapped = textwrap.wrap(
            body,
            width=max(width - len(indent) - len(bullet_prefix), 20),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        lines = [indent + bullet_prefix + wrapped[0]]
        continuation_indent = indent + " " * len(bullet_prefix)
        lines.extend(continuation_indent + part for part in wrapped[1:])
    else:
        wrapped = textwrap.wrap(
            stripped,
            width=max(width - len(indent), 20),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        lines = [indent + part for part in wrapped]

    return [colorize(line, ANSWER_TEXT) for line in lines]


def terminal_text_width() -> int:
    terminal_width = os.get_terminal_size().columns if sys.stdout.isatty() else DEFAULT_WRAP_WIDTH
    return max(min(terminal_width, DEFAULT_WRAP_WIDTH), MIN_WRAP_WIDTH)


def content_width() -> int:
    return max(terminal_text_width() - 2, 40)


def text_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def list_prefix(line: str) -> str:
    match = re.match(r"^([-*+]|\d+[.)])\s+", line)
    return match.group(0) if match else ""


def render_code_block(lines: list[str], language: str) -> list[str]:
    if not use_color():
        return lines

    number_width = len(str(max(len(lines), 1)))
    max_code_width = max([len(line) for line in lines] + [len(language), 8])
    available_width = terminal_text_width()
    code_width = max(
        min(max_code_width, available_width - number_width - 7),
        24,
    )
    inner_width = number_width + 3 + code_width
    title = f" {language or 'code'} "
    top = (
        f"{CODE_BORDER}┌"
        f"{colorize(title, ANSWER_MUTED)}"
        f"{CODE_BORDER}"
        f"{'─' * max(inner_width - len(title), 0)}"
        f"┐{RESET}"
    )
    bottom = f"{CODE_BORDER}└{'─' * inner_width}┘{RESET}"
    number_width = len(str(max(len(lines), 1)))

    rendered = [top]
    for index, line in enumerate(lines, start=1):
        number = f"{index:>{number_width}}"
        highlighted = pad_ansi(truncate_ansi(highlight_code(line), code_width), code_width)
        rendered.append(
            f"{CODE_BORDER}│{RESET}"
            f" {colorize(number, ANSWER_MUTED)} "
            f"{highlighted}"
            f"{CODE_BORDER}│{RESET}"
        )
    rendered.append(bottom)
    return rendered


def highlight_code(line: str) -> str:
    if not line:
        return ""

    code, comment = split_comment(line)
    parts = re.split(r"((?:'[^']*')|(?:\"[^\"]*\"))", code)
    highlighted_parts: list[str] = []

    token_pattern = re.compile(
        r"\b(" + "|".join(sorted(CODE_KEYWORDS)) + r")\b"
        r"|\b\d+(\.\d+)?\b"
    )
    for part in parts:
        if not part:
            continue
        if (part.startswith('"') and part.endswith('"')) or (
            part.startswith("'") and part.endswith("'")
        ):
            highlighted_parts.append(colorize(part, CODE_STRING))
            continue

        highlighted_parts.append(highlight_code_tokens(part, token_pattern))

    highlighted = "".join(highlighted_parts)
    if comment:
        highlighted += colorize(comment, ANSWER_MUTED)
    return highlighted


def highlight_code_tokens(text: str, token_pattern: re.Pattern[str]) -> str:
    highlighted: list[str] = []
    cursor = 0

    for match in token_pattern.finditer(text):
        if match.start() > cursor:
            highlighted.append(colorize(text[cursor : match.start()], CODE_TEXT))

        token = match.group(0)
        if token in CODE_KEYWORDS:
            highlighted.append(colorize(token, CODE_KEYWORD + BOLD))
        else:
            highlighted.append(colorize(token, CODE_NUMBER))
        cursor = match.end()

    if cursor < len(text):
        highlighted.append(colorize(text[cursor:], CODE_TEXT))

    return "".join(highlighted)


def split_comment(line: str) -> tuple[str, str]:
    in_single = False
    in_double = False
    index = 0

    while index < len(line):
        char = line[index]
        previous = line[index - 1] if index else ""

        if char == "'" and not in_double and previous != "\\":
            in_single = not in_single
        elif char == '"' and not in_single and previous != "\\":
            in_double = not in_double
        elif not in_single and not in_double:
            if line.startswith("//", index):
                return line[:index], line[index:]
            if char == "#":
                return line[:index], line[index:]

        index += 1

    return line, ""


def shorten_text(text: str, width: int) -> str:
    normalized = " ".join(str(text).split())
    if not normalized:
        return "—"
    return textwrap.shorten(normalized, width=width, placeholder="…")


def prompt() -> str:
    if not use_color():
        return "> "
    return ACCENT + "> " + USER_INPUT


def colorize(text: str, color: str) -> str:
    if not use_color():
        return text
    return f"{color}{text}{RESET}"


def reset_terminal_color() -> None:
    if use_color():
        print(RESET, end="", flush=True)


def use_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def truncate_ansi(text: str, max_width: int) -> str:
    plain = strip_ansi(text)
    if len(plain) <= max_width:
        return text

    target_width = max(max_width - 1, 0)
    result: list[str] = []
    visible_width = 0
    index = 0

    while index < len(text) and visible_width < target_width:
        if text[index] == "\033":
            match = re.match(r"\033\[[0-9;]*m", text[index:])
            if match:
                result.append(match.group(0))
                index += len(match.group(0))
                continue

        result.append(text[index])
        visible_width += 1
        index += 1

    result.append("…")
    result.append(RESET)
    return "".join(result)


def pad_ansi(text: str, width: int) -> str:
    padding = max(width - len(strip_ansi(text)), 0)
    return text + (" " * padding)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default

    try:
        return float(value)
    except ValueError:
        return default


if __name__ == "__main__":
    main()
