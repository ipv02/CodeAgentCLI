from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
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

from code_agent_cli.agent import APIRequestError, CodeAgent, MissingAPIKeyError


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_GREEN = "\033[92m"
DEFAULT_MAX_FILE_BYTES = 120 * 1024

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
    print(colorize("Code Agent CLI", BOLD + BLUE))
    print(colorize("Команды: /help, /status, /reset, /exit", DIM))
    print(colorize(session_summary(agent), DIM))

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

        if text == "/status":
            print_status(agent)
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

    try:
        with loader("Думаю"):
            answer = agent.send_message(
                payload.request_text,
                history_text=payload.history_text,
            )
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
    print_agent_answer(
        """
Использование:
  code-agent
  code-agent "объясни, чем struct отличается от class в Swift"
  code-agent --file Sources/App.swift "найди ошибки"
  code-agent --file Sources/App.swift --range 40:120 "проверь этот участок"

Переменная окружения:
  export DEEPSEEK_API_KEY="ваш_ключ"

Интерактивные команды:
  /help   показать помощь
  /status показать настройки текущей сессии
  /reset  очистить историю текущей сессии
  /exit   выйти
""".strip()
    )


def print_status(agent: CodeAgent) -> None:
    status = agent.status()
    api_key_status = "задан" if status["api_key_configured"] else "не задан"
    history = f"{status['history_messages']} / {status['max_history_messages']}"

    print_agent_answer(
        f"""
Модель: {status["model"]}
Temperature: {status["temperature"]}
История: {history} сообщений
API URL: {status["api_url"]}
DEEPSEEK_API_KEY: {api_key_status}
""".strip()
    )


def session_summary(agent: CodeAgent) -> str:
    status = agent.status()
    return (
        f"Модель: {status['model']} · "
        f"История: {status['history_messages']}/{status['max_history_messages']}"
    )


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

    answer = input(colorize(message, YELLOW)).strip().lower()
    reset_terminal_color()
    return answer in {"y", "yes", "д", "да"}


def print_agent_answer(answer: str) -> None:
    for line in render_answer(answer):
        print(line)


def render_answer(answer: str) -> list[str]:
    if not use_color():
        return answer.splitlines()

    rendered: list[str] = []
    code_lines: list[str] = []
    code_language = ""
    in_code = False

    for line in answer.splitlines():
        if line.startswith("```"):
            if in_code:
                rendered.extend(render_code_block(code_lines, code_language))
                code_lines = []
                code_language = ""
                in_code = False
            else:
                in_code = True
                code_language = line.removeprefix("```").strip()
            continue

        if in_code:
            code_lines.append(line)
        else:
            rendered.append(colorize(line, BLUE) if line else "")

    if in_code:
        rendered.extend(render_code_block(code_lines, code_language))

    return rendered


def render_code_block(lines: list[str], language: str) -> list[str]:
    width = max([len(strip_ansi(line)) for line in lines] + [len(language), 8])
    width = min(width, 100)
    title = f" {language} " if language else " code "
    top = f"{BRIGHT_CYAN}┌{title}{'─' * max(width - len(title), 0)}┐{RESET}"
    bottom = f"{BRIGHT_CYAN}└{'─' * width}┘{RESET}"
    number_width = len(str(max(len(lines), 1)))

    rendered = [top]
    for index, line in enumerate(lines, start=1):
        number = f"{index:>{number_width}}"
        highlighted = highlight_code(line)
        rendered.append(
            f"{BRIGHT_CYAN}│{RESET} "
            f"{DIM}{number}{RESET} "
            f"{highlighted}"
        )
    rendered.append(bottom)
    return rendered


def highlight_code(line: str) -> str:
    if not line:
        return ""

    code, comment = split_comment(line)
    parts = re.split(r"((?:'[^']*')|(?:\"[^\"]*\"))", code)
    highlighted_parts: list[str] = []

    keyword_pattern = r"\b(" + "|".join(sorted(CODE_KEYWORDS)) + r")\b"
    for part in parts:
        if not part:
            continue
        if (part.startswith('"') and part.endswith('"')) or (
            part.startswith("'") and part.endswith("'")
        ):
            highlighted_parts.append(colorize(part, YELLOW))
            continue

        part = re.sub(
            r"\b\d+(\.\d+)?\b",
            lambda match: colorize(match.group(0), BRIGHT_GREEN),
            part,
        )
        part = re.sub(
            keyword_pattern,
            lambda match: colorize(match.group(0), MAGENTA + BOLD),
            part,
        )
        highlighted_parts.append(part)

    highlighted = "".join(highlighted_parts)
    if comment:
        highlighted += colorize(comment, DIM)
    return highlighted


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


def prompt() -> str:
    if not use_color():
        return "> "
    return GREEN + "> "


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
