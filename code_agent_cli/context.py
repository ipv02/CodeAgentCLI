from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from code_agent_cli.tokens import TokenCounter


SUMMARY_SYSTEM_PROMPT = """
Краткая память предыдущего диалога.
Используй её как контекст, но последние сообщения имеют приоритет.
""".strip()

SUMMARY_UPDATE_PROMPT = """
Обнови summary диалога для code assistant.

Требования к summary:
- сохрани важные решения, требования пользователя, ограничения, названия файлов и текущие договорённости;
- убери повторы и неважные детали;
- пиши структурированно и кратко;
- не добавляй фактов, которых не было в диалоге;
- summary должно помогать продолжить работу без полной старой истории.

Текущее summary:
{summary}

Новые сообщения для сжатия:
{messages}
""".strip()


@dataclass(frozen=True)
class CompressionConfig:
    enabled: bool
    recent_messages: int
    summary_max_tokens: int


@dataclass(frozen=True)
class CompressionStats:
    compressed_messages: int = 0
    summary_tokens_before: int = 0
    summary_tokens_after: int = 0
    prompt_tokens_before: int = 0
    prompt_tokens_after: int = 0

    @property
    def saved_prompt_tokens(self) -> int:
        return max(self.prompt_tokens_before - self.prompt_tokens_after, 0)

    def to_dict(self) -> dict[str, int]:
        return {
            "compressed_messages": self.compressed_messages,
            "summary_tokens_before": self.summary_tokens_before,
            "summary_tokens_after": self.summary_tokens_after,
            "prompt_tokens_before": self.prompt_tokens_before,
            "prompt_tokens_after": self.prompt_tokens_after,
            "saved_prompt_tokens": self.saved_prompt_tokens,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "CompressionStats":
        if not isinstance(value, dict):
            return cls()
        return cls(
            compressed_messages=int(value.get("compressed_messages") or 0),
            summary_tokens_before=int(value.get("summary_tokens_before") or 0),
            summary_tokens_after=int(value.get("summary_tokens_after") or 0),
            prompt_tokens_before=int(value.get("prompt_tokens_before") or 0),
            prompt_tokens_after=int(value.get("prompt_tokens_after") or 0),
        )


def summary_message(summary: str) -> dict[str, str] | None:
    if not summary.strip():
        return None
    return {
        "role": "system",
        "content": f"{SUMMARY_SYSTEM_PROMPT}\n\n{summary.strip()}",
    }


def build_request_messages(
    system_message: dict[str, str],
    summary: str,
    recent_messages: list[dict[str, str]],
    user_message: dict[str, str],
    request_text: str,
) -> list[dict[str, str]]:
    messages = [system_message]
    summary_context = summary_message(summary)
    if summary_context is not None:
        messages.append(summary_context)
    messages.extend(recent_messages)
    messages.append(user_message)
    if user_message["content"] != request_text:
        messages[-1] = {
            "role": "user",
            "content": request_text,
        }
    return messages


def split_messages_for_compression(
    messages: list[dict[str, str]],
    config: CompressionConfig,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not config.enabled:
        return [], messages
    if len(messages) <= config.recent_messages:
        return [], messages

    compress_count = len(messages) - config.recent_messages
    return messages[:compress_count], messages[compress_count:]


def build_summary_update_messages(
    system_prompt: str,
    summary: str,
    messages_to_compress: list[dict[str, str]],
    max_summary_tokens: int,
) -> list[dict[str, str]]:
    formatted_messages = format_messages(messages_to_compress)
    request = SUMMARY_UPDATE_PROMPT.format(
        summary=summary.strip() or "(summary пока пустое)",
        messages=formatted_messages,
    )
    request += f"\n\nЖелаемый максимум summary: {max_summary_tokens} токенов."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]


def compression_stats(
    token_counter: TokenCounter,
    system_message: dict[str, str],
    summary_before: str,
    summary_after: str,
    messages_before: list[dict[str, str]],
    messages_after: list[dict[str, str]],
    compressed_count: int,
) -> CompressionStats:
    before_context = [system_message, *messages_before]
    before_summary = summary_message(summary_before)
    if before_summary is not None:
        before_context.insert(1, before_summary)

    after_context = [system_message, *messages_after]
    after_summary = summary_message(summary_after)
    if after_summary is not None:
        after_context.insert(1, after_summary)

    return CompressionStats(
        compressed_messages=compressed_count,
        summary_tokens_before=token_counter.count_text(summary_before),
        summary_tokens_after=token_counter.count_text(summary_after),
        prompt_tokens_before=token_counter.count_messages(before_context),
        prompt_tokens_after=token_counter.count_messages(after_context),
    )


def format_messages(messages: list[dict[str, str]]) -> str:
    formatted: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        formatted.append(f"{index}. {role}:\n{content}")
    return "\n\n".join(formatted)
