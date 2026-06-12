from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


SLIDING_STRATEGY = "sliding"
FACTS_STRATEGY = "facts"
BRANCHING_STRATEGY = "branching"
CONTEXT_STRATEGIES = {SLIDING_STRATEGY, FACTS_STRATEGY, BRANCHING_STRATEGY}
DEFAULT_BRANCH = "main"

FACTS_SYSTEM_PROMPT = """
Ты обновляешь sticky facts для code assistant.
Верни только JSON object с ключами и строковыми значениями.
Не добавляй markdown.
""".strip()

FACTS_UPDATE_PROMPT = """
Обнови facts на основе нового сообщения пользователя.

Facts должны хранить только устойчиво важную информацию:
- goal: цель пользователя или проекта;
- constraints: ограничения и правила;
- preferences: предпочтения пользователя;
- decisions: принятые решения;
- current_task: текущая задача;
- files: важные файлы или модули;
- risks: важные риски или edge cases.

Текущие facts:
{facts}

Новое сообщение пользователя:
{message}

Верни обновленный JSON object. Удаляй устаревшее, сохраняй важное.
""".strip()


@dataclass
class BranchState:
    messages: list[dict[str, str]]
    facts: dict[str, str] = field(default_factory=dict)
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "facts": self.facts,
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BranchState":
        messages = value.get("messages")
        facts = value.get("facts")
        checkpoints = value.get("checkpoints")
        return cls(
            messages=messages if isinstance(messages, list) else [],
            facts=normalize_facts(facts),
            checkpoints=checkpoints if isinstance(checkpoints, dict) else {},
        )


def normalize_strategy(value: str | None) -> str:
    strategy = (value or FACTS_STRATEGY).strip().lower()
    return strategy if strategy in CONTEXT_STRATEGIES else FACTS_STRATEGY


def normalize_facts(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    facts: dict[str, str] = {}
    for key, fact_value in value.items():
        if not isinstance(key, str):
            continue
        if fact_value is None:
            continue
        facts[key] = str(fact_value)
    return facts


def facts_message(facts: dict[str, str]) -> dict[str, str] | None:
    if not facts:
        return None
    lines = ["Sticky facts / key-value memory:"]
    for key in sorted(facts):
        value = facts[key].strip()
        if value:
            lines.append(f"- {key}: {value}")
    if len(lines) == 1:
        return None
    return {
        "role": "system",
        "content": "\n".join(lines),
    }


def build_request_messages(
    system_message: dict[str, str],
    facts: dict[str, str],
    recent_messages: list[dict[str, str]],
    user_message: dict[str, str],
    request_text: str,
) -> list[dict[str, str]]:
    messages = [system_message]
    facts_context = facts_message(facts)
    if facts_context is not None:
        messages.append(facts_context)
    messages.extend(recent_messages)
    messages.append(user_message)
    if user_message["content"] != request_text:
        messages[-1] = {
            "role": "user",
            "content": request_text,
        }
    return messages


def trim_visible_messages(
    messages: list[dict[str, str]],
    max_messages: int,
) -> list[dict[str, str]]:
    if max_messages < 1:
        return []
    return messages[-max_messages:]


def build_facts_update_messages(
    current_facts: dict[str, str],
    user_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FACTS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": FACTS_UPDATE_PROMPT.format(
                facts=json.dumps(current_facts, ensure_ascii=False, indent=2),
                message=user_text,
            ),
        },
    ]


def parse_facts_response(text: str) -> dict[str, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("facts response is not a JSON object")
    return normalize_facts(payload)


def checkpoint_state(branch: BranchState) -> dict[str, Any]:
    return {
        "messages": [dict(message) for message in branch.messages],
        "facts": dict(branch.facts),
    }


def branch_from_checkpoint(checkpoint: dict[str, Any]) -> BranchState:
    return BranchState(
        messages=[dict(message) for message in checkpoint.get("messages", [])],
        facts=normalize_facts(checkpoint.get("facts")),
        checkpoints={},
    )
