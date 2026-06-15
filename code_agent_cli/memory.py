from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


WORKING_MEMORY_KEYS = {
    "current_task",
    "plan",
    "files",
    "risks",
    "temporary_constraints",
    "state",
}
LONG_TERM_MEMORY_KEYS = {
    "profile",
    "preferences",
    "project_decisions",
    "stable_constraints",
    "knowledge",
    "decisions",
}

MEMORY_ROUTER_SYSTEM_PROMPT = """
Ты обновляешь явную layered memory для code assistant.
Верни только JSON object. Не добавляй markdown.
""".strip()

MEMORY_ROUTER_PROMPT = """
Разложи новое сообщение пользователя по слоям памяти.

Слои:
- working: данные текущей задачи, которые могут измениться или исчезнуть после завершения задачи.
- long_term: устойчивый профиль, предпочтения, решения проекта и знания, полезные между запусками.
- discard: одноразовые детали, приветствия, шум и данные, которые не нужно хранить.

Правила:
- short-term memory хранится отдельно как история сообщений, не возвращай ее в JSON.
- Обновляй только то, что явно следует из сообщения.
- Не сохраняй полный код или большие файлы. Сохраняй только краткие ссылки на файлы/решения.
- Если сообщение не содержит данных для слоя, верни пустой object для этого слоя.
- Значения в working и long_term должны быть строками.

Текущая память:
{memory}

Новое сообщение пользователя:
{message}

Верни JSON строго в формате:
{{
  "working": {{}},
  "long_term": {{}},
  "discard": []
}}
""".strip()


@dataclass
class MemoryState:
    working: dict[str, str] = field(default_factory=dict)
    long_term: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "working": self.working,
            "long_term": self.long_term,
        }

    def to_history_dict(self) -> dict[str, Any]:
        return {
            "working": self.working,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MemoryState":
        if not isinstance(value, dict):
            return cls()
        return cls(
            working=normalize_memory_layer(value.get("working")),
            long_term=normalize_memory_layer(value.get("long_term")),
        )

    @classmethod
    def from_legacy_facts(cls, facts: dict[str, str]) -> "MemoryState":
        working: dict[str, str] = {}
        long_term: dict[str, str] = {}
        for key, value in facts.items():
            if key in WORKING_MEMORY_KEYS or key in {"goal"}:
                working[key] = value
            elif key in LONG_TERM_MEMORY_KEYS or key in {"constraints"}:
                long_term[key] = value
            else:
                working[key] = value
        return cls(working=working, long_term=long_term)

    def apply_update(self, update: "MemoryUpdate") -> None:
        merge_layer(self.working, update.working)
        merge_layer(self.long_term, update.long_term)

    def clear_working(self) -> None:
        self.working.clear()

    def clear_long_term(self) -> None:
        self.long_term.clear()

    def combined(self) -> dict[str, str]:
        return {
            **self.long_term,
            **self.working,
        }


@dataclass(frozen=True)
class MemoryUpdate:
    working: dict[str, str]
    long_term: dict[str, str]
    discard: list[str] = field(default_factory=list)


def normalize_memory_layer(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, layer_value in value.items():
        if not isinstance(key, str) or layer_value is None:
            continue
        normalized[key] = str(layer_value).strip()
    return {
        key: value
        for key, value in normalized.items()
        if value
    }


def merge_layer(target: dict[str, str], update: dict[str, str]) -> None:
    for key, value in update.items():
        normalized_value = value.strip()
        if normalized_value:
            target[key] = normalized_value
        else:
            target.pop(key, None)


def memory_messages(memory: MemoryState) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    long_term_message = memory_layer_message("Long-term memory", memory.long_term)
    working_message = memory_layer_message("Working memory", memory.working)
    if long_term_message is not None:
        messages.append(long_term_message)
    if working_message is not None:
        messages.append(working_message)
    return messages


def memory_layer_message(title: str, layer: dict[str, str]) -> dict[str, str] | None:
    if not layer:
        return None
    lines = [f"{title}:"]
    for key in sorted(layer):
        value = layer[key].strip()
        if value:
            lines.append(f"- {key}: {value}")
    if len(lines) == 1:
        return None
    return {
        "role": "system",
        "content": "\n".join(lines),
    }


def build_memory_update_messages(
    current_memory: MemoryState,
    user_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MEMORY_ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": MEMORY_ROUTER_PROMPT.format(
                memory=json.dumps(current_memory.to_dict(), ensure_ascii=False, indent=2),
                message=user_text,
            ),
        },
    ]


def parse_memory_update_response(text: str) -> MemoryUpdate:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("memory update response is not a JSON object")

    discard_payload = payload.get("discard")
    discard = [
        str(item)
        for item in discard_payload
        if item is not None
    ] if isinstance(discard_payload, list) else []

    return MemoryUpdate(
        working=normalize_memory_layer(payload.get("working")),
        long_term=normalize_memory_layer(payload.get("long_term")),
        discard=discard,
    )


def default_profile_file() -> Path:
    configured_path = os.getenv("CODE_AGENT_PROFILE_FILE")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path.home() / ".code-agent-cli" / "profile.md"


class ProfileStorage:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_profile_file()

    def load(self) -> dict[str, str]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError:
            return {}

        return parse_profile_markdown(text)

    def save(self, long_term: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(render_profile_markdown(long_term), encoding="utf-8")
        temp_path.replace(self.path)


def parse_profile_markdown(text: str) -> dict[str, str]:
    profile: dict[str, str] = {}
    current_key = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key:
            value = "\n".join(current_lines).strip()
            if value:
                profile[current_key] = value
        current_key = ""
        current_lines = []

    for line in text.splitlines():
        match = re.match(r"^##\s+([A-Za-z0-9_-]+)\s*$", line)
        if match:
            flush()
            current_key = match.group(1)
            continue
        if current_key:
            current_lines.append(line)

    flush()
    return normalize_memory_layer(profile)


def render_profile_markdown(long_term: dict[str, str]) -> str:
    lines = [
        "# CodeAgentCLI Profile",
        "",
        "Long-term memory used by code-agent.",
        "Edit this file carefully: each section title is a memory key.",
        "",
    ]
    for key in sorted(long_term):
        value = long_term[key].strip()
        if not value:
            continue
        lines.extend((f"## {key}", "", value, ""))
    return "\n".join(lines).rstrip() + "\n"
