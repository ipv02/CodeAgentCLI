from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


HISTORY_VERSION = 1


def default_history_file() -> Path:
    configured_path = os.getenv("CODE_AGENT_HISTORY_FILE")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path.home() / ".code-agent-cli" / "history.json"


class HistoryStorage:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_history_file()

    def load(self) -> list[dict[str, str]] | None:
        try:
            raw_payload = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None

        normalized_messages = [normalize_message(message) for message in messages]
        valid_messages = [message for message in normalized_messages if message is not None]
        return valid_messages or None

    def save(self, messages: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": HISTORY_VERSION,
            "messages": messages,
        }
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)


def normalize_message(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None

    role = value.get("role")
    content = value.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        return None

    if role not in {"system", "user", "assistant"}:
        return None

    return {
        "role": role,
        "content": content,
    }
