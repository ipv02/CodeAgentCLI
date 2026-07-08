from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LOCAL_MODEL = "llama3.2:3b"


class LocalLLMError(Exception):
    """Base error for local LLM chat."""


class LocalLLMConnectionError(LocalLLMError):
    """Raised when Ollama is not reachable."""


class LocalLLMRequestError(LocalLLMError):
    """Raised when Ollama returns an API error."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


@dataclass
class LocalLLMChatService:
    model: str = field(
        default_factory=lambda: os.getenv("CODE_AGENT_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
    )
    ollama_url: str = field(
        default_factory=lambda: os.getenv("CODE_AGENT_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    )
    timeout: float = field(
        default_factory=lambda: env_float("CODE_AGENT_LOCAL_TIMEOUT", 120.0)
    )
    temperature: float = field(
        default_factory=lambda: env_float("CODE_AGENT_LOCAL_TEMPERATURE", 0.2)
    )
    max_history_messages: int = field(
        default_factory=lambda: env_int("CODE_AGENT_LOCAL_MAX_HISTORY", 20)
    )

    def __post_init__(self) -> None:
        self.ollama_url = self.ollama_url.rstrip("/")
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "Ты локальный ассистент внутри CodeAgentCLI. "
                    "Отвечай полезно, кратко и явно отмечай ограничения, если не уверен."
                ),
            }
        ]

    def send(self, text: str) -> str:
        user_message = {"role": "user", "content": text}
        request_messages = [*self.messages, user_message]
        answer = self.generate(request_messages)

        self.messages = [*request_messages, {"role": "assistant", "content": answer}]
        self._trim_history()
        return answer

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        response_payload = self._post_json("/api/chat", payload)
        message = response_payload.get("message")
        if not isinstance(message, dict):
            raise LocalLLMError("Ollama вернула ответ без message.")
        answer = str(message.get("content") or "").strip()
        if not answer:
            answer = "Ollama вернула пустой ответ."
        return answer

    def reset(self) -> None:
        self.messages = self.messages[:1]

    def ping(self) -> dict[str, Any]:
        return self._post_json("/api/show", {"model": self.model})

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.ollama_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_text = response.read().decode("utf-8")
        except HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")
            raise LocalLLMRequestError(
                error.code,
                format_ollama_error(error.code, response_text, self.model),
            ) from error
        except URLError as error:
            raise LocalLLMConnectionError(
                f"Ollama недоступна на {self.ollama_url}. Запустите: ollama serve"
            ) from error
        except TimeoutError as error:
            raise LocalLLMConnectionError(
                f"Ollama не ответила за {self.timeout:.0f} сек."
            ) from error

        try:
            value = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise LocalLLMError(f"Ollama вернула некорректный JSON: {error}") from error
        if not isinstance(value, dict):
            raise LocalLLMError("Ollama вернула JSON не в формате object.")
        return value

    def _trim_history(self) -> None:
        if len(self.messages) <= self.max_history_messages + 1:
            return
        self.messages = [self.messages[0], *self.messages[-self.max_history_messages :]]


def format_ollama_error(status_code: int, response_text: str, model: str) -> str:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        payload = {}

    message = payload.get("error") if isinstance(payload, dict) else ""
    if not isinstance(message, str) or not message:
        message = response_text.strip() or f"HTTP {status_code}"

    if status_code == 404 and "not found" in message.lower():
        return f"Модель {model} не найдена. Выполните: ollama pull {model}"
    return f"Ollama вернула HTTP {status_code}: {message}"


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
