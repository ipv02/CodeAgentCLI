from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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
from code_agent_cli.tokens import TokenBreakdown


RESET = "\033[0m"
BOLD = "\033[1m"
MUTED = "\033[38;5;245m"
SUBTLE = "\033[38;5;239m"
ACCENT = "\033[38;5;81m"
ACCENT_SOFT = "\033[38;5;110m"
USER_INPUT = "\033[38;5;214m"
SUCCESS = "\033[38;5;114m"
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    agent = CodeAgent()

    if args.prompt:
        prompt = build_prompt(args)
        if not send(agent, prompt):
            raise SystemExit(1)
        return

    run_interactive_session(agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-agent",
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
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.line_range and args.file is None:
        parser.error("--range можно использовать только вместе с --file.")

    if args.force_file and args.file is None:
        parser.error("--force-file можно использовать только вместе с --file.")

    if args.max_file_bytes < 1:
        parser.error("--max-file-bytes должен быть положительным числом.")


def run_interactive_session(agent: CodeAgent) -> None:
    print(colorize("Code Agent CLI", BOLD + ACCENT))
    print(colorize("Команды: /help, /status, /tokens, /context, /branch, /reset, /exit", MUTED))
    print(colorize(session_summary(agent), MUTED))

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
            agent.reset_history()
            print("История очищена.")
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

        if command == "/branch":
            handle_branch_command(agent, argument)
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
    payload = normalize_prompt(prompt)
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


def normalize_prompt(prompt: str | PromptPayload) -> PromptPayload:
    if isinstance(prompt, PromptPayload):
        return prompt
    return PromptPayload(request_text=prompt)


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
            "code-agent",
            'code-agent "объясни, чем struct отличается от class в Swift"',
            'code-agent --file Sources/App.swift "найди ошибки"',
            'code-agent --file Sources/App.swift --range 40:120 "проверь этот участок"',
        ),
    )
    print()
    print_section(
        "Переменная окружения",
        ('export DEEPSEEK_API_KEY="ваш_ключ"',),
    )
    print()
    print_command_help(
        (
            ("/help", "показать помощь"),
            ("/status", "показать настройки текущей сессии"),
            ("/tokens", "показать токены истории и последнего запроса"),
            ("/tokens текст", "посчитать токены текста без отправки в API"),
            ("/context", "показать стратегию контекста и facts"),
            ("/context strategy NAME", "переключить: sliding, facts, branching"),
            ("/branch list", "показать ветки"),
            ("/branch compare A B", "сравнить две ветки"),
            ("/branch checkpoint NAME", "сохранить checkpoint активной ветки"),
            ("/branch create NAME [CHECKPOINT]", "создать ветку"),
            ("/branch switch NAME", "переключиться на ветку"),
            ("/reset", "очистить сохраненную историю"),
            ("/exit", "выйти"),
        )
    )


def print_section(title: str, lines: tuple[str, ...]) -> None:
    print(header_line(title))
    for line in lines:
        print(command_line(line))


def print_command_help(commands: tuple[tuple[str, str], ...]) -> None:
    print(header_line("Интерактивные команды"))
    width = max(len(command) for command, _ in commands)
    for command, description in commands:
        if not use_color():
            print(f"  {command:<{width}}  {description}")
            continue
        print(f"  {COMMAND}{command:<{width}}{RESET}  {MUTED}{description}{RESET}")


def header_line(text: str) -> str:
    if not use_color():
        return f"{text}:"
    return f"{ACCENT}{BOLD}{text}{RESET}{SUBTLE}:{RESET}"


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
        status_line("Facts", f"{status['facts_count']} ключей"),
        status_line("Facts tokens", str(status["facts_tokens"])),
        status_line("Facts max", f"{status['facts_max_tokens']} токенов"),
    ):
        print(line)

    if status["last_memory_error"]:
        print(status_line("Ошибка memory", str(status["last_memory_error"]), WARNING))

    print()
    print(header_line("История"))
    for line in (
        status_line("Сообщения", f"{history} сообщений"),
        status_line("Файл истории", str(status["history_file"])),
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


def session_summary(agent: CodeAgent) -> str:
    status = agent.status()
    return (
        f"Модель: {status['model']} · "
        f"Стратегия: {status['context_strategy']} · "
        f"Ветка: {status['active_branch']} · "
        f"История: {status['history_messages']}/{status['max_history_messages']} · "
        f"Контекст: {status['current_history_tokens']}/{status['context_limit']} токенов · "
        f"{'загружена' if status['history_loaded'] else 'новая'}"
    )


def print_current_token_state(agent: CodeAgent, request_text: str | None = None) -> None:
    status = agent.status()
    print(header_line("Текущие токены"))
    for line in (
        status_line("Вся история диалога", str(status["current_history_tokens"])),
        status_line("Лимит модели", str(status["context_limit"])),
        status_line("Остаток", str(status["remaining_context_tokens"])),
        status_line("Стратегия", str(status["context_strategy"])),
        status_line("Активная ветка", str(status["active_branch"])),
        status_line("Facts tokens", str(status["facts_tokens"])),
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
    if report["facts"]:
        print(status_line("Facts", f"{report['facts_tokens']} токенов"))
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

    print("Использование: /context или /context strategy sliding|facts|branching")


def print_context_report(agent: CodeAgent) -> None:
    report = agent.context_report()
    print(header_line("Контекст"))
    for line in (
        status_line("Стратегия", str(report["strategy"])),
        status_line("Активная ветка", str(report["active_branch"])),
        status_line("Сообщения", f"{report['messages']} / {report['max_messages']}"),
        status_line("Prompt tokens", str(report["prompt_tokens_current_strategy"])),
        status_line("Sliding tokens", str(report["prompt_tokens_sliding"])),
        status_line("Facts tokens", str(report["facts_tokens"])),
    ):
        print(line)

    if report["last_memory_error"]:
        print(status_line("Ошибка memory", str(report["last_memory_error"]), WARNING))

    facts = report["facts"]
    if facts:
        print()
        print(header_line("Facts"))
        for key in sorted(facts):
            print(status_line(key, facts[key]))


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


def print_branch_report(agent: CodeAgent) -> None:
    report = agent.branch_report()
    print(header_line("Ветки"))
    print(status_line("Активная", str(report["active_branch"]), SUCCESS))
    for name, branch in report["branches"].items():
        marker = "*" if name == report["active_branch"] else " "
        value = (
            f"{branch['messages']} сообщений, "
            f"{branch['facts']} facts, "
            f"{branch['prompt_tokens']} prompt tokens, "
            f"{len(branch['checkpoints'])} checkpoints"
        )
        print(status_line(f"{marker} {name}", value))
        if branch["last_user"]:
            print(status_line("  последний user", shorten_text(branch["last_user"], 88), MUTED))
        if branch["current_task"]:
            print(status_line("  current_task", shorten_text(branch["current_task"], 88), MUTED))
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

    facts_diff = report["facts_diff"]
    if facts_diff:
        print()
        print(header_line("Разные facts"))
        for key, diff in facts_diff.items():
            print(status_line(f"{key} / {report['left_name']}", shorten_text(diff["left"], 110), MUTED))
            print(status_line(f"{key} / {report['right_name']}", shorten_text(diff["right"], 110), MUTED))
    else:
        print()
        print(status_line("Facts", "одинаковые", WARNING))


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


if __name__ == "__main__":
    main()
