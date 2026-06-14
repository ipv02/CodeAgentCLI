from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from code_agent_cli.context import (
    BRANCHING_STRATEGY,
    DEFAULT_BRANCH,
    FACTS_STRATEGY,
    BranchState,
    branch_from_checkpoint,
    build_facts_update_messages,
    build_request_messages,
    checkpoint_state,
    normalize_strategy,
    parse_facts_response,
    trim_visible_messages,
)
from code_agent_cli.storage import HistoryStorage, default_history_file
from code_agent_cli.tokens import ModelTokenConfig, TokenBreakdown, TokenCounter


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


class ContextLimitExceededError(CodeAgentError):
    """Raised when a request would exceed the configured context limit."""

    def __init__(self, breakdown: TokenBreakdown) -> None:
        self.breakdown = breakdown
        super().__init__(
            "Запрос не отправлен: превышен лимит контекстного окна."
        )


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
    context_strategy: str = field(
        default_factory=lambda: normalize_strategy(os.getenv("CODE_AGENT_CONTEXT_STRATEGY"))
    )
    facts_max_tokens: int = field(
        default_factory=lambda: env_int("CODE_AGENT_FACTS_MAX_TOKENS", 1200)
    )
    context_limit: int = field(
        default_factory=lambda: env_int("CODE_AGENT_CONTEXT_LIMIT", 64_000)
    )
    input_price_per_1m: float = field(
        default_factory=lambda: env_float("CODE_AGENT_INPUT_PRICE_PER_1M", 0.28)
    )
    output_price_per_1m: float = field(
        default_factory=lambda: env_float("CODE_AGENT_OUTPUT_PRICE_PER_1M", 0.42)
    )
    temperature: float = field(default_factory=lambda: env_float("CODE_AGENT_TEMPERATURE", 0.2))
    history_file: Path = field(default_factory=default_history_file)

    def __post_init__(self) -> None:
        self.token_usage = TokenUsage()
        self.token_counter = TokenCounter(
            self.model,
            ModelTokenConfig(
                context_limit=self.context_limit,
                input_price_per_1m=self.input_price_per_1m,
                output_price_per_1m=self.output_price_per_1m,
            ),
        )
        self.last_token_breakdown: TokenBreakdown | None = None
        self.last_actual_usage: dict[str, Any] = {}
        self.last_memory_error = ""
        self.history_storage = HistoryStorage(self.history_file)
        self.history_loaded = False
        self.branches: dict[str, BranchState] = {}
        self.active_branch = DEFAULT_BRANCH
        self.messages = self._load_history()
        self._trim_history_if_needed()

    def send_message(self, text: str, history_text: str | None = None) -> str:
        user_message = self._message("user", history_text or text)
        request_messages = self._request_messages_for_user_message(user_message, text)
        self.last_token_breakdown = self.token_counter.build_breakdown(
            request_messages,
            text,
        )
        self.last_actual_usage = {}

        if not self.last_token_breakdown.fits_context:
            raise ContextLimitExceededError(self.last_token_breakdown)

        if not self.api_key:
            raise MissingAPIKeyError(
                'Не задан DEEPSEEK_API_KEY. Выполните: export DEEPSEEK_API_KEY="ваш_ключ"'
            )

        self._update_facts_if_needed(text)
        request_messages = self._request_messages_for_user_message(user_message, text)
        self.last_token_breakdown = self.token_counter.build_breakdown(
            request_messages,
            text,
        )
        if not self.last_token_breakdown.fits_context:
            raise ContextLimitExceededError(self.last_token_breakdown)

        answer, usage = self._perform_request(request_messages)

        self.last_actual_usage = usage
        self.token_usage.add(usage)
        self.messages = [
            *self.messages,
            user_message,
            self._message("assistant", answer),
        ]
        self._trim_history_if_needed()
        self._save_history()
        return answer

    def status(self) -> dict[str, str | int | float | bool]:
        current_history_tokens = self.token_counter.count_messages(
            self._messages_with_memory(self.messages)
        )
        facts_tokens = (
            self.token_counter.count_text(json.dumps(self.facts, ensure_ascii=False))
            if self.facts
            else 0
        )
        return {
            "api_key_configured": bool(self.api_key),
            "api_url": self.api_url,
            "model": self.model,
            "temperature": self.temperature,
            "context_limit": self.context_limit,
            "current_history_tokens": current_history_tokens,
            "remaining_context_tokens": self.context_limit - current_history_tokens,
            "history_messages": max(len(self.messages) - 1, 0),
            "max_history_messages": self.max_history_messages,
            "context_strategy": self.context_strategy,
            "active_branch": self.active_branch,
            "branch_count": len(self.branches),
            "facts_count": len(self.facts),
            "facts_tokens": facts_tokens,
            "facts_max_tokens": self.facts_max_tokens,
            "last_memory_error": self.last_memory_error,
            "session_total_tokens": self.token_usage.total_tokens,
            "session_prompt_tokens": self.token_usage.prompt_tokens,
            "session_completion_tokens": self.token_usage.completion_tokens,
            "history_file": str(self.history_storage.path),
            "history_loaded": self.history_loaded,
        }

    def estimate_tokens(self, text: str, history_text: str | None = None) -> TokenBreakdown:
        user_message = self._message("user", history_text or text)
        request_messages = self._request_messages_for_user_message(user_message, text)
        return self.token_counter.build_breakdown(request_messages, text)

    def reset_history(self) -> None:
        self.branches = {
            DEFAULT_BRANCH: BranchState(messages=self._initial_messages())
        }
        self.active_branch = DEFAULT_BRANCH
        self.messages = self.branches[self.active_branch].messages
        self.last_memory_error = ""
        self.history_loaded = False
        self._save_history()

    @property
    def facts(self) -> dict[str, str]:
        return self.branches[self.active_branch].facts

    def _perform_request(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
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
        self.messages = self._trimmed_messages(self.messages)
        self.branches[self.active_branch].messages = self.messages

    def _trimmed_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(messages) <= self.max_history_messages + 1:
            return messages

        system_message = messages[0]
        recent_messages = messages[-self.max_history_messages:]
        return [system_message, *recent_messages]

    def _load_history(self) -> list[dict[str, str]]:
        state = self.history_storage.load()
        if state is None:
            self.branches = {
                DEFAULT_BRANCH: BranchState(messages=self._initial_messages())
            }
            self.active_branch = DEFAULT_BRANCH
            return self.branches[self.active_branch].messages

        self.context_strategy = normalize_strategy(state.strategy or self.context_strategy)
        self.branches = state.branches
        self.active_branch = state.active_branch
        self.history_loaded = True
        self._ensure_active_branch()
        return self.branches[self.active_branch].messages

    def _normalize_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        user_visible_messages = [
            message
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
        return [self._message("system", SYSTEM_PROMPT), *user_visible_messages]

    def _save_history(self) -> None:
        self.branches[self.active_branch].messages = self.messages
        self.history_storage.save(
            self.context_strategy,
            self.active_branch,
            self.branches,
        )

    def _initial_messages(self) -> list[dict[str, str]]:
        return [self._message("system", SYSTEM_PROMPT)]

    def _request_messages_for_user_message(
        self,
        history_user_message: dict[str, str],
        request_text: str,
    ) -> list[dict[str, str]]:
        recent_messages = trim_visible_messages(
            self.messages[1:],
            self.max_history_messages,
        )
        facts = self.facts if self.context_strategy in {FACTS_STRATEGY, BRANCHING_STRATEGY} else {}
        return build_request_messages(
            self.messages[0],
            facts,
            recent_messages,
            history_user_message,
            request_text,
        )

    def _messages_with_memory(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        facts = self.facts if self.context_strategy in {FACTS_STRATEGY, BRANCHING_STRATEGY} else {}
        return build_request_messages(
            messages[0],
            facts,
            trim_visible_messages(messages[1:], self.max_history_messages),
            self._message("user", ""),
            "",
        )[:-1]

    def _update_facts_if_needed(self, user_text: str) -> None:
        if self.context_strategy not in {FACTS_STRATEGY, BRANCHING_STRATEGY}:
            return

        try:
            facts_text, usage = self._perform_request(
                build_facts_update_messages(self.facts, user_text),
                max_tokens=self.facts_max_tokens,
            )
            self.branches[self.active_branch].facts = parse_facts_response(facts_text)
            self.token_usage.add(usage)
            self.last_memory_error = ""
        except Exception as error:
            self.last_memory_error = str(error)

    def _ensure_active_branch(self) -> None:
        if not self.branches:
            self.branches = {
                DEFAULT_BRANCH: BranchState(messages=self._initial_messages())
            }
        if self.active_branch not in self.branches:
            self.active_branch = next(iter(self.branches))
        branch = self.branches[self.active_branch]
        branch.messages = self._normalize_history(branch.messages)

    def set_context_strategy(self, strategy: str) -> None:
        self.context_strategy = normalize_strategy(strategy)
        self._save_history()

    def branch_list(self) -> list[str]:
        return sorted(self.branches)

    def create_checkpoint(self, name: str) -> None:
        branch = self.branches[self.active_branch]
        branch.checkpoints[name] = checkpoint_state(branch)
        self._save_history()

    def create_branch(self, name: str, checkpoint: str | None = None) -> None:
        if name in self.branches:
            raise CodeAgentError(f"Ветка уже существует: {name}")
        source = self.branches[self.active_branch]
        if checkpoint:
            checkpoint_payload = source.checkpoints.get(checkpoint)
            if checkpoint_payload is None:
                raise CodeAgentError(f"Checkpoint не найден: {checkpoint}")
            self.branches[name] = branch_from_checkpoint(checkpoint_payload)
        else:
            self.branches[name] = BranchState(
                messages=[dict(message) for message in source.messages],
                facts=dict(source.facts),
                checkpoints={},
            )
        self._save_history()

    def switch_branch(self, name: str) -> None:
        if name not in self.branches:
            raise CodeAgentError(f"Ветка не найдена: {name}")
        self.branches[self.active_branch].messages = self.messages
        self.active_branch = name
        self.messages = self.branches[self.active_branch].messages
        self._save_history()

    def delete_branch(self, name: str) -> None:
        if name == self.active_branch:
            raise CodeAgentError("Нельзя удалить активную ветку.")
        if name not in self.branches:
            raise CodeAgentError(f"Ветка не найдена: {name}")
        del self.branches[name]
        self._save_history()

    def context_report(self) -> dict[str, Any]:
        request_messages = self._messages_with_memory(self.messages)
        sliding_messages = [
            self.messages[0],
            *trim_visible_messages(self.messages[1:], self.max_history_messages),
        ]
        return {
            "strategy": self.context_strategy,
            "active_branch": self.active_branch,
            "branches": self.branch_list(),
            "facts": dict(self.facts),
            "facts_tokens": (
                self.token_counter.count_text(json.dumps(self.facts, ensure_ascii=False))
                if self.facts
                else 0
            ),
            "prompt_tokens_current_strategy": self.token_counter.count_messages(request_messages),
            "prompt_tokens_sliding": self.token_counter.count_messages(sliding_messages),
            "messages": max(len(self.messages) - 1, 0),
            "max_messages": self.max_history_messages,
            "last_memory_error": self.last_memory_error,
        }

    def branch_report(self) -> dict[str, Any]:
        active = self.branches[self.active_branch]
        return {
            "active_branch": self.active_branch,
            "branches": {
                name: self._branch_summary(branch)
                for name, branch in sorted(self.branches.items())
            },
            "active_checkpoints": sorted(active.checkpoints),
        }

    def compare_branches(self, left_name: str, right_name: str) -> dict[str, Any]:
        if left_name not in self.branches:
            raise CodeAgentError(f"Ветка не найдена: {left_name}")
        if right_name not in self.branches:
            raise CodeAgentError(f"Ветка не найдена: {right_name}")

        left = self.branches[left_name]
        right = self.branches[right_name]
        return {
            "left_name": left_name,
            "right_name": right_name,
            "left": self._branch_summary(left),
            "right": self._branch_summary(right),
            "facts_diff": self._facts_diff(left.facts, right.facts),
        }

    def _branch_summary(self, branch: BranchState) -> dict[str, Any]:
        request_messages = self._messages_with_branch_memory(branch)
        return {
            "messages": max(len(branch.messages) - 1, 0),
            "facts": len(branch.facts),
            "facts_values": dict(branch.facts),
            "checkpoints": sorted(branch.checkpoints),
            "prompt_tokens": self.token_counter.count_messages(request_messages),
            "last_user": self._last_message_content(branch.messages, "user"),
            "last_assistant": self._last_message_content(branch.messages, "assistant"),
            "current_task": branch.facts.get("current_task", ""),
            "goal": branch.facts.get("goal", ""),
        }

    def _messages_with_branch_memory(self, branch: BranchState) -> list[dict[str, str]]:
        facts = branch.facts if self.context_strategy in {FACTS_STRATEGY, BRANCHING_STRATEGY} else {}
        return build_request_messages(
            branch.messages[0],
            facts,
            trim_visible_messages(branch.messages[1:], self.max_history_messages),
            self._message("user", ""),
            "",
        )[:-1]

    @staticmethod
    def _last_message_content(messages: list[dict[str, str]], role: str) -> str:
        for message in reversed(messages):
            if message.get("role") == role:
                return message.get("content", "")
        return ""

    @staticmethod
    def _facts_diff(
        left: dict[str, str],
        right: dict[str, str],
    ) -> dict[str, dict[str, str]]:
        diff: dict[str, dict[str, str]] = {}
        for key in sorted(set(left) | set(right)):
            left_value = left.get(key, "")
            right_value = right.get(key, "")
            if left_value != right_value:
                diff[key] = {
                    "left": left_value,
                    "right": right_value,
                }
        return diff

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
