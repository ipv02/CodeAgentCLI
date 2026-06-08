from __future__ import annotations

import argparse
import sys
from pathlib import Path

from code_agent_cli.agent import CodeAgent, MissingAPIKeyError


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
    print("Code Agent CLI")
    print("Команды: /help, /reset, /exit")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
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
        answer = agent.send_message(prompt)
    except MissingAPIKeyError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print()
    print(answer)
    print()


def print_help() -> None:
    print(
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


if __name__ == "__main__":
    main()
