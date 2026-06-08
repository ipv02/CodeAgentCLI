from __future__ import annotations

import json
import os
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


class APIRequestError(CodeAgentError):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


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


@dataclass
class CodeAgent:
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    api_url: str = field(
        default_factory=lambda: os.getenv(
            "CODE_AGENT_API_URL",
            "https://api.deepseek.com/chat/completions",
        )
    )
    model: str = field(default_factory=lambda: os.getenv("CODE_AGENT_MODEL", "deepseek-v4-flash"))
    max_history_messages: int = field(
        default_factory=lambda: env_int("CODE_AGENT_MAX_HISTORY", 20)
    )
    temperature: float = field(default_factory=lambda: env_float("CODE_AGENT_TEMPERATURE", 0.2))

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

    def status(self) -> dict[str, str | int | float | bool]:
        return {
            "api_key_configured": bool(self.api_key),
            "api_url": self.api_url,
            "model": self.model,
            "temperature": self.temperature,
            "history_messages": max(len(self.messages) - 1, 0),
            "max_history_messages": self.max_history_messages,
        }

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
            raise APIRequestError(error.code, format_api_error(error.code, response_text)) from error

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


def format_api_error(status_code: int, response_text: str) -> str:
    detail = extract_api_error_message(response_text)

    if status_code == 401:
        return f"Ошибка API 401: проверьте DEEPSEEK_API_KEY. {detail}".strip()
    if status_code == 429:
        return f"Ошибка API 429: лимит запросов или квоты. Попробуйте позже. {detail}".strip()
    if status_code >= 500:
        return f"Ошибка API {status_code}: сервис временно недоступен. {detail}".strip()
    return f"Ошибка API {status_code}: {detail or response_text or 'нет деталей'}"


def extract_api_error_message(response_text: str) -> str:
    if not response_text:
        return ""

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        return str(message) if message else ""

    return response_text
