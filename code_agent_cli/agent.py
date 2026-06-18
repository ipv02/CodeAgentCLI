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
    MEMORY_STRATEGY,
    BranchState,
    branch_from_checkpoint,
    build_request_messages,
    checkpoint_state,
    normalize_strategy,
    trim_visible_messages,
)
from code_agent_cli.invariants import (
    InvariantAgent,
    InvariantStorage,
    default_invariants_file,
)
from code_agent_cli.memory import (
    MemoryState,
    ProfileStorage,
    default_profile_file,
)
from code_agent_cli.storage import HistoryStorage, default_history_file
from code_agent_cli.subagents import (
    SYSTEM_PROMPT,
    MemoryAgent,
    ResponseAgent,
    TaskStateAgent,
)
from code_agent_cli.tokens import ModelTokenConfig, TokenBreakdown, TokenCounter


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


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "да"}


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
    memory_max_tokens: int = field(
        default_factory=lambda: env_int("CODE_AGENT_MEMORY_MAX_TOKENS", 1200)
    )
    auto_memory_updates: bool = field(
        default_factory=lambda: env_bool("CODE_AGENT_AUTO_MEMORY", True)
    )
    auto_task_state_updates: bool = field(
        default_factory=lambda: env_bool("CODE_AGENT_AUTO_TASK_STATE", True)
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
    profile_file: Path = field(default_factory=default_profile_file)
    invariants_file: Path = field(default_factory=default_invariants_file)

    def __post_init__(self) -> None:
        self.history_file = Path(self.history_file)
        self.profile_file = Path(self.profile_file)
        self.invariants_file = Path(self.invariants_file)
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
        self.last_invariant_error = ""
        self.history_storage = HistoryStorage(self.history_file)
        self.profile_storage = ProfileStorage(self.profile_file)
        self.invariant_storage = InvariantStorage(self.invariants_file)
        self.response_agent = ResponseAgent()
        self.memory_agent = MemoryAgent(max_tokens=self.memory_max_tokens)
        self.task_state_agent = TaskStateAgent()
        self.invariant_agent = InvariantAgent()
        self.history_loaded = False
        self.invariants = self.invariant_storage.load()
        self.branches: dict[str, BranchState] = {}
        self.active_branch = DEFAULT_BRANCH
        self.messages = self._load_history()
        self._load_profile_memory()
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

        conflict_answer = self._refuse_if_heuristic_invariant_conflict(text)
        if conflict_answer is not None:
            self._save_answer_to_history(user_message, conflict_answer)
            return conflict_answer

        if not self.api_key:
            raise MissingAPIKeyError(
                'Не задан DEEPSEEK_API_KEY. Выполните: export DEEPSEEK_API_KEY="ваш_ключ"'
            )

        conflict_answer = self._refuse_if_invariant_conflict(text)
        if conflict_answer is not None:
            self._save_answer_to_history(user_message, conflict_answer)
            return conflict_answer

        self._update_memory_if_needed(text)
        self._update_task_state_before_request(text)
        request_messages = self._request_messages_for_user_message(user_message, text)
        self.last_token_breakdown = self.token_counter.build_breakdown(
            request_messages,
            text,
        )
        if not self.last_token_breakdown.fits_context:
            raise ContextLimitExceededError(self.last_token_breakdown)

        answer, usage = self.response_agent.run(self._perform_request, request_messages)

        self.last_actual_usage = usage
        self.token_usage.add(usage)
        self._update_task_state_after_answer(text, answer)
        self.messages = [
            *self.messages,
            user_message,
            self._message("assistant", answer),
        ]
        self._trim_history_if_needed()
        self._save_history()
        return answer

    def handle_memory_only_message(self, text: str, history_text: str | None = None) -> str | None:
        if self.context_strategy not in {MEMORY_STRATEGY, BRANCHING_STRATEGY}:
            return None
        if not self.auto_memory_updates:
            return None
        if not self.memory_agent.is_memory_only_intent(text):
            return None

        before_working = dict(self.memory.working)
        before_long_term = dict(self.memory.long_term)
        self._update_memory_if_needed(text)
        self._update_task_state_before_request(text)

        working_changed = before_working != self.memory.working
        long_term_changed = before_long_term != self.memory.long_term
        if long_term_changed and working_changed:
            answer = "Память обновлена: профиль и данные текущей задачи сохранены."
        elif long_term_changed:
            answer = "Профиль обновлен."
        elif working_changed:
            answer = "Рабочая память обновлена."
        elif self.last_memory_error:
            answer = "Не удалось обновить память автоматически."
        else:
            answer = "Новых данных для памяти не найдено."

        user_message = self._message("user", history_text or text)
        self.messages = [
            *self.messages,
            user_message,
            self._message("assistant", answer),
        ]
        self._trim_history_if_needed()
        self.last_token_breakdown = None
        self.last_actual_usage = {}
        self._save_history()
        return answer

    def status(self) -> dict[str, str | int | float | bool]:
        current_history_tokens = self.token_counter.count_messages(
            self._messages_with_memory(self.messages)
        )
        memory_tokens = self._memory_tokens(self.memory)
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
            "working_memory_count": len(self.memory.working),
            "long_term_memory_count": len(self.memory.long_term),
            "invariant_count": len(self.invariants),
            "task_stage": self.memory.task_state.stage,
            "task_current_step": self.memory.task_state.current_step,
            "task_expected_action": self.memory.task_state.expected_action,
            "memory_tokens": memory_tokens,
            "memory_count": len(self.memory.working) + len(self.memory.long_term),
            "memory_max_tokens": self.memory_max_tokens,
            "auto_memory_updates": self.auto_memory_updates,
            "auto_task_state_updates": self.auto_task_state_updates,
            "last_memory_error": self.last_memory_error,
            "last_invariant_error": self.last_invariant_error,
            "session_total_tokens": self.token_usage.total_tokens,
            "session_prompt_tokens": self.token_usage.prompt_tokens,
            "session_completion_tokens": self.token_usage.completion_tokens,
            "history_file": str(self.history_storage.path),
            "profile_file": str(self.profile_storage.path),
            "invariants_file": str(self.invariant_storage.path),
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
        self.memory.long_term = self.profile_storage.load()
        self.last_memory_error = ""
        self.last_invariant_error = ""
        self.history_loaded = False
        self._save_history()

    def reset_agent(self) -> None:
        self.profile_storage.save({})
        self.token_usage = TokenUsage()
        self.last_token_breakdown = None
        self.last_actual_usage = {}
        self.last_invariant_error = ""
        self.reset_history()

    @property
    def memory(self) -> MemoryState:
        return self.branches[self.active_branch].memory

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
        return self.response_agent.normalize_history(messages)

    def _save_history(self) -> None:
        self.branches[self.active_branch].messages = self.messages
        self.history_storage.save(
            self.context_strategy,
            self.active_branch,
            self.branches,
        )

    def _initial_messages(self) -> list[dict[str, str]]:
        return self.response_agent.initial_messages()

    def _request_messages_for_user_message(
        self,
        history_user_message: dict[str, str],
        request_text: str,
    ) -> list[dict[str, str]]:
        recent_messages = trim_visible_messages(
            self.messages[1:],
            self.max_history_messages,
        )
        memory = self.memory if self.context_strategy in {MEMORY_STRATEGY, BRANCHING_STRATEGY} else MemoryState()
        request_messages = self.response_agent.build_request_messages(
            self.messages[0],
            memory,
            recent_messages,
            history_user_message,
            request_text,
        )
        return self._with_invariant_messages(request_messages)

    def _messages_with_memory(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        memory = self.memory if self.context_strategy in {MEMORY_STRATEGY, BRANCHING_STRATEGY} else MemoryState()
        request_messages = self.response_agent.build_request_messages(
            messages[0],
            memory,
            trim_visible_messages(messages[1:], self.max_history_messages),
            self._message("user", ""),
            "",
        )
        return self._with_invariant_messages(request_messages)[:-1]

    def _save_answer_to_history(
        self,
        user_message: dict[str, str],
        answer: str,
    ) -> None:
        self.messages = [
            *self.messages,
            user_message,
            self._message("assistant", answer),
        ]
        self._trim_history_if_needed()
        self._save_history()

    def _with_invariant_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        invariant_messages = self.invariant_agent.build_policy_messages(self.invariants)
        if not invariant_messages:
            return messages
        return [messages[0], *invariant_messages, *messages[1:]]

    def _refuse_if_invariant_conflict(self, user_text: str) -> str | None:
        if not self.invariants:
            return None

        try:
            result, usage = self.invariant_agent.check_conflict(
                self._perform_request,
                self.invariants,
                user_text,
            )
            self.token_usage.add(usage)
            self.last_invariant_error = ""
        except Exception as error:
            self.last_invariant_error = str(error)
            result = self.invariant_agent.check_conflict_with_heuristics(
                self.invariants,
                user_text,
            )

        if not result.conflict:
            return None

        return self._build_invariant_refusal(result)

    def _refuse_if_heuristic_invariant_conflict(self, user_text: str) -> str | None:
        if not self.invariants:
            return None

        result = self.invariant_agent.check_conflict_with_heuristics(
            self.invariants,
            user_text,
        )
        if not result.conflict:
            return None

        return self._build_invariant_refusal(result)

    def _build_invariant_refusal(self, result: Any) -> str:
        violated = result.violated_invariants or self.invariants
        lines = [
            "Не могу предложить решение в таком виде: запрос конфликтует с обязательными инвариантами.",
            "",
            "Нарушаемые инварианты:",
            *[f"- {invariant}" for invariant in violated],
        ]
        if result.explanation:
            lines.extend(("", f"Причина: {result.explanation}"))
        if result.safe_alternative:
            lines.extend(("", f"Безопасная альтернатива: {result.safe_alternative}"))
        else:
            lines.extend(("", "Могу помочь переформулировать задачу так, чтобы сохранить эти ограничения."))
        return "\n".join(lines)

    def _update_memory_if_needed(self, user_text: str) -> None:
        if self.context_strategy not in {MEMORY_STRATEGY, BRANCHING_STRATEGY}:
            return
        if not self.auto_memory_updates:
            return

        try:
            working, long_term, _discard, usage = self.memory_agent.run(
                self._perform_request,
                self.memory,
                user_text,
            )
            self._apply_memory_update(user_text, working, long_term)
            self.token_usage.add(usage)
            self.last_memory_error = ""
        except Exception as error:
            self.last_memory_error = str(error)
            fallback_working, fallback_long_term = self.memory_agent.enrich_with_fallback(
                user_text,
                {},
                {},
            )
            if fallback_working or fallback_long_term:
                self._apply_memory_update(user_text, fallback_working, fallback_long_term)

    def _apply_memory_update(
        self,
        user_text: str,
        working: dict[str, str],
        long_term: dict[str, str],
    ) -> None:
        del user_text
        self.branches[self.active_branch].memory.working.update(working)
        for key, value in working.items():
            if not value.strip():
                self.branches[self.active_branch].memory.working.pop(key, None)
        self.branches[self.active_branch].memory.long_term.update(long_term)
        for key, value in long_term.items():
            if not value.strip():
                self.branches[self.active_branch].memory.long_term.pop(key, None)
        if long_term:
            self._sync_long_term_memory()
            self.profile_storage.save(self.memory.long_term)

    def _update_task_state_before_request(self, user_text: str) -> None:
        if not self.auto_task_state_updates:
            return
        if self.task_state_agent.prepare(self.memory.task_state, user_text):
            self._save_history()

    def _update_task_state_after_answer(self, user_text: str, answer: str) -> None:
        if not self.auto_task_state_updates:
            return
        if self.memory.task_state.stage == "paused":
            return

        try:
            update, usage = self.task_state_agent.run(
                self._perform_request,
                self.memory.task_state,
                user_text,
                answer,
            )
            self.memory.task_state.merge_update(update)
            if not self.memory.task_state.current_step:
                self.memory.task_state.current_step = user_text.strip()
            if not self.memory.task_state.summary:
                self.memory.task_state.summary = user_text.strip()
            if not self.memory.task_state.expected_action:
                self.task_state_agent.apply_fallback(self.memory.task_state, user_text, answer)
                self._save_history()
                return
            self.token_usage.add(usage)
            self._save_history()
        except Exception:
            self._update_task_state_with_heuristics(user_text, answer)

    def _update_task_state_with_heuristics(self, user_text: str, answer: str) -> None:
        self.task_state_agent.apply_fallback(self.memory.task_state, user_text, answer)
        self._save_history()

    def _ensure_active_branch(self) -> None:
        if not self.branches:
            self.branches = {
                DEFAULT_BRANCH: BranchState(messages=self._initial_messages())
            }
        if self.active_branch not in self.branches:
            self.active_branch = next(iter(self.branches))
        branch = self.branches[self.active_branch]
        branch.messages = self._normalize_history(branch.messages)

    def _load_profile_memory(self) -> None:
        profile_memory = self.profile_storage.load()
        had_history_long_term = any(
            branch.memory.long_term
            for branch in self.branches.values()
        )
        for branch in self.branches.values():
            branch.memory.long_term = dict(profile_memory)
        if had_history_long_term:
            self._save_history()

    def _sync_long_term_memory(self) -> None:
        long_term = dict(self.memory.long_term)
        for branch in self.branches.values():
            branch.memory.long_term = dict(long_term)

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
                memory=MemoryState.from_dict(source.memory.to_dict()),
                checkpoints={},
            )
        self._save_history()

    def switch_branch(self, name: str) -> None:
        if name not in self.branches:
            raise CodeAgentError(f"Ветка не найдена: {name}")
        self.branches[self.active_branch].messages = self.messages
        self.active_branch = name
        self.messages = self.branches[self.active_branch].messages
        self.branches[self.active_branch].memory.long_term = dict(
            self.profile_storage.load()
        )
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
            "memory": self.memory.to_dict(),
            "working_memory": dict(self.memory.working),
            "long_term_memory": dict(self.memory.long_term),
            "invariants": list(self.invariants),
            "invariant_tokens": self._invariant_tokens(),
            "task_state": self.memory.task_state.to_dict(),
            "memory_tokens": self._memory_tokens(self.memory),
            "prompt_tokens_current_strategy": self.token_counter.count_messages(request_messages),
            "prompt_tokens_sliding": self.token_counter.count_messages(sliding_messages),
            "messages": max(len(self.messages) - 1, 0),
            "max_messages": self.max_history_messages,
            "last_memory_error": self.last_memory_error,
            "last_invariant_error": self.last_invariant_error,
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
            "memory_diff": self._memory_diff(
                left.memory.combined(),
                right.memory.combined(),
            ),
        }

    def _branch_summary(self, branch: BranchState) -> dict[str, Any]:
        request_messages = self._messages_with_branch_memory(branch)
        return {
            "messages": max(len(branch.messages) - 1, 0),
            "memory_count": len(branch.memory.combined()),
            "memory": branch.memory.to_dict(),
            "working_memory": dict(branch.memory.working),
            "long_term_memory": dict(branch.memory.long_term),
            "task_state": branch.memory.task_state.to_dict(),
            "checkpoints": sorted(branch.checkpoints),
            "prompt_tokens": self.token_counter.count_messages(request_messages),
            "last_user": self._last_message_content(branch.messages, "user"),
            "last_assistant": self._last_message_content(branch.messages, "assistant"),
            "current_task": branch.memory.working.get("current_task", ""),
            "goal": branch.memory.working.get("goal", ""),
        }

    def _messages_with_branch_memory(self, branch: BranchState) -> list[dict[str, str]]:
        memory = branch.memory if self.context_strategy in {MEMORY_STRATEGY, BRANCHING_STRATEGY} else MemoryState()
        request_messages = build_request_messages(
            branch.messages[0],
            memory,
            trim_visible_messages(branch.messages[1:], self.max_history_messages),
            self._message("user", ""),
            "",
        )
        return self._with_invariant_messages(request_messages)[:-1]

    def clear_working_memory(self) -> None:
        self.memory.clear_working()
        self._save_history()

    def clear_short_term_memory(self) -> None:
        self.messages = self._initial_messages()
        self.branches[self.active_branch].messages = self.messages
        self._save_history()

    def clear_long_term_memory(self) -> None:
        self.memory.clear_long_term()
        self._sync_long_term_memory()
        self.profile_storage.save(self.memory.long_term)
        self._save_history()

    def clear_all_memory(self) -> None:
        self.clear_short_term_memory()
        self.clear_working_memory()
        self.clear_long_term_memory()

    def invariants_report(self) -> dict[str, Any]:
        return {
            "path": str(self.invariant_storage.path),
            "invariants": list(self.invariants),
            "count": len(self.invariants),
            "tokens": self._invariant_tokens(),
        }

    def add_invariant(self, value: str) -> None:
        invariant = value.strip()
        if not invariant:
            raise CodeAgentError("Инвариант не должен быть пустым.")
        if invariant not in self.invariants:
            self.invariants.append(invariant)
            self.invariant_storage.save(self.invariants)

    def delete_invariant(self, index: int) -> str:
        if index < 1 or index > len(self.invariants):
            raise CodeAgentError("Инвариант с таким номером не найден.")
        removed = self.invariants.pop(index - 1)
        self.invariant_storage.save(self.invariants)
        return removed

    def clear_invariants(self) -> None:
        self.invariants = []
        self.invariant_storage.save(self.invariants)

    def task_report(self) -> dict[str, str]:
        return self.memory.task_state.to_dict()

    def set_task_stage(self, stage: str) -> None:
        if not self.task_state_agent.set_stage(self.memory.task_state, stage):
            raise CodeAgentError(
                "Этап задачи должен быть planning, execution, validation, done или paused."
            )
        self._save_history()

    def set_task_current_step(self, value: str) -> None:
        self.memory.task_state.current_step = value.strip()
        self._save_history()

    def set_task_expected_action(self, value: str) -> None:
        self.memory.task_state.expected_action = value.strip()
        self._save_history()

    def set_task_summary(self, value: str) -> None:
        self.memory.task_state.summary = value.strip()
        self._save_history()

    def pause_task(self) -> None:
        self.memory.task_state.pause()
        self._save_history()

    def resume_task(self) -> None:
        self.memory.task_state.resume()
        self._save_history()

    def clear_task_state(self) -> None:
        self.memory.task_state.clear()
        self._save_history()

    @staticmethod
    def _last_message_content(messages: list[dict[str, str]], role: str) -> str:
        for message in reversed(messages):
            if message.get("role") == role:
                return message.get("content", "")
        return ""

    @staticmethod
    def _memory_diff(
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

    def _memory_tokens(self, memory: MemoryState) -> int:
        if not memory.working and not memory.long_term:
            return 0
        memory_only_messages = build_request_messages(
            self._message("system", ""),
            memory,
            [],
            self._message("user", ""),
            "",
        )[1:-1]
        return self.token_counter.count_messages(memory_only_messages)

    def _invariant_tokens(self) -> int:
        invariant_messages = self.invariant_agent.build_policy_messages(self.invariants)
        if not invariant_messages:
            return 0
        return self.token_counter.count_messages(invariant_messages)

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
