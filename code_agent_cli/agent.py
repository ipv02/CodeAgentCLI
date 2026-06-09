from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from code_agent_cli.storage import HistoryStorage, default_history_file


SYSTEM_PROMPT = """
Ты профессиональный senior software engineer и code assistant.

Твоя экспертиза:
- backend-разработка, API, базы данных, архитектура, безопасность, производительность;
- frontend-разработка, UI, состояние приложения, интеграции, сборка;
- mobile-разработка, iOS, Android, Swift, Kotlin, React Native;
- нейросети, LLM API, RAG, tool calling, prompt engineering;
- agentic coding, CLI-инструменты, автоматизация разработки, code review.

Помогай пользователю писать, объяснять, улучшать, отлаживать и проектировать код.
Отвечай практично, структурированно и без лишней воды.
Если пользователь просит исправить код, кратко объясни проблему и дай исправленный вариант.
Если есть несколько подходов, назови лучший по умолчанию и коротко объясни trade-off.
Если данных недостаточно и без них легко ошибиться, задай уточняющий вопрос.
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


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: dict[str, Any]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)


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
    history_file: Path = field(default_factory=default_history_file)

    def __post_init__(self) -> None:
        self.token_usage = TokenUsage()
        self.history_storage = HistoryStorage(self.history_file)
        self.history_loaded = False
        self.messages = self._load_history()
        self._trim_history_if_needed()

    def send_message(self, text: str, history_text: str | None = None) -> str:
        if not self.api_key:
            raise MissingAPIKeyError(
                'Не задан DEEPSEEK_API_KEY. Выполните: export DEEPSEEK_API_KEY="ваш_ключ"'
            )

        user_message = self._message("user", history_text or text)
        self.messages.append(user_message)
        self._trim_history_if_needed()

        try:
            request_messages = self._request_messages(user_message, text)
            answer, usage = self._perform_request(request_messages)
        except Exception:
            self.messages = [message for message in self.messages if message != user_message]
            raise

        self.token_usage.add(usage)
        self.messages.append(self._message("assistant", answer))
        self._trim_history_if_needed()
        self._save_history()
        return answer

    def status(self) -> dict[str, str | int | float | bool]:
        return {
            "api_key_configured": bool(self.api_key),
            "api_url": self.api_url,
            "model": self.model,
            "temperature": self.temperature,
            "history_messages": max(len(self.messages) - 1, 0),
            "max_history_messages": self.max_history_messages,
            "session_total_tokens": self.token_usage.total_tokens,
            "session_prompt_tokens": self.token_usage.prompt_tokens,
            "session_completion_tokens": self.token_usage.completion_tokens,
            "history_file": str(self.history_storage.path),
            "history_loaded": self.history_loaded,
        }

    def reset_history(self) -> None:
        self.messages = self._initial_messages()
        self.history_loaded = False
        self._save_history()

    def _perform_request(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
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
        usage = response_payload.get("usage") or {}
        if not choices:
            return response_text or "Нет ответа", usage

        message = choices[0].get("message") or {}
        content = message.get("content")
        return content or response_text or "Нет ответа", usage

    def _trim_history_if_needed(self) -> None:
        if len(self.messages) <= self.max_history_messages + 1:
            return

        system_message = self.messages[0]
        recent_messages = self.messages[-self.max_history_messages:]
        self.messages = [system_message, *recent_messages]

    def _load_history(self) -> list[dict[str, str]]:
        messages = self.history_storage.load()
        if messages is None:
            return self._initial_messages()

        self.history_loaded = True
        return self._normalize_history(messages)

    def _normalize_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        user_visible_messages = [
            message
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
        return [self._message("system", SYSTEM_PROMPT), *user_visible_messages]

    def _save_history(self) -> None:
        self.history_storage.save(self.messages)

    def _initial_messages(self) -> list[dict[str, str]]:
        return [self._message("system", SYSTEM_PROMPT)]

    def _request_messages(
        self,
        history_user_message: dict[str, str],
        request_text: str,
    ) -> list[dict[str, str]]:
        if history_user_message["content"] == request_text:
            return self.messages

        request_messages = list(self.messages)
        request_messages[-1] = self._message("user", request_text)
        return request_messages

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
