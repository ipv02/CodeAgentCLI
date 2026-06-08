from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from code_agent_cli.agent import CodeAgent, MissingAPIKeyError


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_GREEN = "\033[92m"

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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = CodeAgent()

    if args.prompt:
        prompt = build_prompt(args.prompt, args.file)
        send(agent, prompt)
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
    return parser


def run_interactive_session(agent: CodeAgent) -> None:
    print(colorize("Code Agent CLI", BOLD + BLUE))
    print(colorize("Команды: /help, /reset, /exit", DIM))

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

        send(agent, text)


def build_prompt(prompt_parts: list[str], file_path: Path | None) -> str:
    user_prompt = " ".join(prompt_parts).strip()

    if file_path is None:
        return user_prompt

    try:
        file_content = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Ошибка: не удалось прочитать файл {file_path}: {error}") from error

    return f"""{user_prompt}

Код из файла {file_path}:
```text
{file_content}
```"""


def send(agent: CodeAgent, prompt: str) -> None:
    try:
        with loader("Думаю"):
            answer = agent.send_message(prompt)
    except MissingAPIKeyError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print()
    print_agent_answer(answer)
    print()


@contextmanager
def loader(label: str) -> Iterator[None]:
    if not use_color():
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

Переменная окружения:
  export DEEPSEEK_API_KEY="ваш_ключ"

Интерактивные команды:
  /help   показать помощь
  /reset  очистить историю текущей сессии
  /exit   выйти
""".strip()
    )


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


if __name__ == "__main__":
    main()
