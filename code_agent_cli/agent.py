from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """
Ты code assistant. Помогай пользователю писать, объяснять, улучшать и отлаживать код.
Отвечай практично, кратко и структурированно.
Если пользователь просит исправить код, объясни проблему и дай исправленный вариант.
Если данных недостаточно, задай уточняющий вопрос.
""".strip()


class CodeAgentError(Exception):
    """Base error for CodeAgentCLI."""


class MissingAPIKeyError(CodeAgentError):
    """Raised when DEEPSEEK_API_KEY is not configured."""


@dataclass
class CodeAgent:
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    api_url: str = "https://api.deepseek.com/chat/completions"
    model: str = "deepseek-v4-flash"
    max_history_messages: int = 20
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self.reset_history()

    def send_message(self, text: str) -> str:
        if not self.api_key:
            raise MissingAPIKeyError(
                'Не задан DEEPSEEK_API_KEY. Выполните: export DEEPSEEK_API_KEY="ваш_ключ"'
            )

        user_message = self._message("user", text)
        self.messages.append(user_message)
        self._trim_history_if_needed()

        try:
            answer = self._perform_request(self.messages)
        except Exception:
            self.messages = [message for message in self.messages if message != user_message]
            raise

        self.messages.append(self._message("assistant", answer))
        self._trim_history_if_needed()
        return answer

    def reset_history(self) -> None:
        self.messages: list[dict[str, str]] = [
            self._message("system", SYSTEM_PROMPT)
        ]

    def _perform_request(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        request = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                response_text = response.read().decode("utf-8")
        except HTTPError as error:
            response_text = error.read().decode("utf-8")
            return response_text or "Ошибка API"

        response_payload: dict[str, Any] = json.loads(response_text)
        choices = response_payload.get("choices") or []
        if not choices:
            return response_text or "Нет ответа"

        message = choices[0].get("message") or {}
        content = message.get("content")
        return content or response_text or "Нет ответа"

    def _trim_history_if_needed(self) -> None:
        if len(self.messages) <= self.max_history_messages + 1:
            return

        system_message = self.messages[0]
        recent_messages = self.messages[-self.max_history_messages:]
        self.messages = [system_message, *recent_messages]

    @staticmethod
    def _message(role: str, content: str) -> dict[str, str]:
        return {
            "role": role,
            "content": content,
        }
