from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from code_agent_cli.context import build_request_messages
from code_agent_cli.memory import (
    MemoryState,
    TaskState,
    TaskStateUpdate,
    TaskTransitionError,
    build_memory_update_messages,
    build_task_state_update_messages,
    normalize_task_stage,
    parse_memory_update_response,
    parse_task_state_update_response,
)


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


RequestFn = Callable[[list[dict[str, str]], Optional[int]], tuple[str, dict[str, Any]]]


@dataclass
class ResponseAgent:
    system_prompt: str = SYSTEM_PROMPT

    def system_message(self) -> dict[str, str]:
        return {"role": "system", "content": self.system_prompt}

    def initial_messages(self) -> list[dict[str, str]]:
        return [self.system_message()]

    def normalize_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        user_visible_messages = [
            message
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
        return [self.system_message(), *user_visible_messages]

    def build_request_messages(
        self,
        system_message: dict[str, str],
        memory: MemoryState,
        recent_messages: list[dict[str, str]],
        history_user_message: dict[str, str],
        request_text: str,
    ) -> list[dict[str, str]]:
        return build_request_messages(
            system_message,
            memory,
            recent_messages,
            history_user_message,
            request_text,
        )

    def run(
        self,
        request_fn: RequestFn,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any]]:
        return request_fn(messages, None)


@dataclass
class MemoryAgent:
    max_tokens: int = 1200
    working_key_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "task": "current_task",
            "currentTask": "current_task",
            "file": "files",
            "risk": "risks",
            "constraint": "temporary_constraints",
        }
    )
    long_term_key_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "role": "user_role",
            "user_profile": "user_role",
            "framework": "preferred_framework",
            "preferred_language": "language_preference",
            "language": "language_preference",
            "architecture": "architecture_preference",
        }
    )
    memory_only_prefixes: tuple[str, ...] = (
        "запомни",
        "запомни,",
        "remember",
        "remember that",
        "сохрани в память",
        "запиши в память",
    )

    def is_memory_only_intent(self, user_text: str) -> bool:
        normalized = user_text.strip().lower()
        return any(normalized.startswith(prefix) for prefix in self.memory_only_prefixes)

    def run(
        self,
        request_fn: RequestFn,
        memory: MemoryState,
        user_text: str,
    ) -> tuple[dict[str, str], dict[str, str], list[str], dict[str, Any]]:
        response_text, usage = request_fn(
            build_memory_update_messages(memory, user_text),
            self.max_tokens,
        )
        update = parse_memory_update_response(response_text)
        return update.working, update.long_term, update.discard, usage

    def enrich_with_fallback(
        self,
        user_text: str,
        working: dict[str, str],
        long_term: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        fallback_working, fallback_long_term = self.extract_fallback(user_text)
        merged_working = self.normalize_working_keys({**fallback_working, **working})
        merged_long_term = self.normalize_long_term_keys({**fallback_long_term, **long_term})
        return merged_working, merged_long_term

    def extract_fallback(self, user_text: str) -> tuple[dict[str, str], dict[str, str]]:
        text = user_text.strip()
        lower = text.lower()
        working: dict[str, str] = {}
        long_term: dict[str, str] = {}

        if any(token in lower for token in ("я ios-разработчик", "я iOS-разработчик".lower(), "ios developer")):
            long_term["user_role"] = "iOS-разработчик"
        if "swiftui" in lower:
            long_term["preferred_framework"] = "SwiftUI"
        if any(token in lower for token in ("на русском", "по-русски", "русском")):
            long_term["language_preference"] = "русский"
        if "production-grade" in lower:
            long_term["architecture_preference"] = "production-grade"

        files_match = re.search(r"(?:рабочие\s+файлы|files?)\s*:\s*(.+)", text, re.IGNORECASE)
        if files_match:
            files_value = re.split(r"(?:\.\s+риск\s*:|\.\s+риски\s*:)", files_match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
            working["files"] = files_value.strip().rstrip(".")

        risks_match = re.search(r"(?:риск|риски)\s*:\s*(.+)", text, re.IGNORECASE)
        if risks_match:
            working["risks"] = risks_match.group(1).strip().rstrip(".")

        constraints_match = re.search(r"(?:ограничения|constraints?)\s*:\s*(.+)", text, re.IGNORECASE)
        if constraints_match:
            working["temporary_constraints"] = constraints_match.group(1).strip().rstrip(".")

        sentence_parts = re.split(r"[.!?]\s+", text)
        for part in sentence_parts:
            candidate = part.strip()
            lower_candidate = candidate.lower()
            if not candidate:
                continue
            if any(token in lower_candidate for token in ("сейчас", "проектируем", "делаем", "задача", "нужно")):
                if "рабочие файлы" in lower_candidate or "риск:" in lower_candidate or "риски:" in lower_candidate:
                    continue
                working["current_task"] = candidate.rstrip(".")
                break

        return working, long_term

    def normalize_working_keys(self, working: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in working.items():
            target_key = self.working_key_aliases.get(key, key)
            normalized[target_key] = value
        return normalized

    def normalize_long_term_keys(self, long_term: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in long_term.items():
            target_key = self.long_term_key_aliases.get(key, key)
            normalized[target_key] = value
        return normalized


@dataclass
class TaskStateAgent:
    max_tokens: int = 300
    continue_commands: set[str] = field(
        default_factory=lambda: {"продолжай", "continue", "resume", "продолжить"}
    )
    pause_commands: set[str] = field(
        default_factory=lambda: {"пауза", "pause"}
    )
    plan_approval_markers: tuple[str, ...] = (
        "план утверждаю",
        "план ок",
        "план подходит",
        "утверждаю план",
        "согласен с планом",
        "согласна с планом",
        "можно приступать",
        "приступай",
        "делай",
        "начинай реализацию",
        "approve plan",
        "plan approved",
        "go ahead",
    )
    implementation_markers: tuple[str, ...] = (
        "реализуй",
        "сделай реализацию",
        "напиши код",
        "внеси изменения",
        "implement",
        "write code",
    )
    validation_markers: tuple[str, ...] = (
        "проверь",
        "прогони тесты",
        "валидация",
        "validate",
        "run tests",
    )
    completion_markers: tuple[str, ...] = (
        "заверши",
        "готово",
        "done",
        "finish",
        "mark done",
    )

    def prepare(self, task_state: TaskState, user_text: str) -> bool:
        normalized = user_text.strip().lower()
        if normalized in self.continue_commands:
            task_state.resume()
            return True
        if normalized in self.pause_commands:
            task_state.pause()
            return True
        if task_state.is_empty:
            normalized_text = user_text.strip()
            task_state.current_step = normalized_text
            task_state.summary = normalized_text
            task_state.stage = "planning"
            task_state.expected_action = "проанализировать задачу и предложить следующий шаг"
            return True
        if task_state.stage == "planning" and self.is_plan_approval_intent(user_text):
            task_state.transition_to("execution")
            task_state.current_step = user_text.strip()
            task_state.expected_action = "выполнить реализацию по утвержденному плану"
            return True
        return False

    def run(
        self,
        request_fn: RequestFn,
        current_state: TaskState,
        user_text: str,
        assistant_text: str,
    ) -> tuple[TaskStateUpdate, dict[str, Any]]:
        response_text, usage = request_fn(
            build_task_state_update_messages(current_state, user_text, assistant_text),
            self.max_tokens,
        )
        return parse_task_state_update_response(response_text), usage

    def apply_fallback(
        self,
        task_state: TaskState,
        user_text: str,
        answer: str,
    ) -> str:
        answer_lower = answer.lower()
        user_lower = user_text.lower()

        if any(token in answer_lower for token in ("провер", "test", "compileall", "валид")):
            desired_stage = "validation"
        elif any(token in answer_lower for token in ("готово", "заверш", "done")):
            desired_stage = "done"
        elif any(token in answer_lower for token in ("план", "шаг", "архитектур")) and not any(
            token in answer_lower for token in ("```", "func ", "class ", "def ")
        ):
            desired_stage = "planning"
        else:
            desired_stage = "execution"

        if (
            task_state.stage == "planning"
            and desired_stage == "execution"
            and not self.is_plan_approval_intent(user_text)
        ):
            transition_error = (
                "Недопустимый переход задачи: planning -> execution. "
                "Сначала пользователь должен явно утвердить план."
            )
        else:
            transition_error = self.apply_stage_transition(task_state, desired_stage)

        if not task_state.summary:
            task_state.summary = user_text.strip()
        if not task_state.current_step:
            task_state.current_step = user_text.strip()

        if transition_error and task_state.stage == "planning" and desired_stage == "execution":
            task_state.expected_action = "дождаться утверждения плана пользователем"
        elif task_state.stage == "planning":
            task_state.expected_action = "дождаться утверждения плана пользователем"
        elif task_state.stage == "execution":
            task_state.expected_action = "продолжить реализацию или уточнить следующий рабочий шаг"
        elif task_state.stage == "validation":
            task_state.expected_action = "проверить результат и подтвердить завершение"
        elif task_state.stage == "done":
            task_state.expected_action = "задача завершена"

        if any(token in user_lower for token in self.continue_commands):
            task_state.resume()

        return transition_error

    def set_stage(self, task_state: TaskState, stage: str) -> bool:
        normalized = normalize_task_stage(stage)
        if not normalized:
            return False
        task_state.transition_to(normalized)
        return True

    def requested_stage_from_user(self, user_text: str) -> str:
        normalized = user_text.strip().lower()
        if self.is_plan_approval_intent(user_text):
            return "execution"
        if self.has_any_marker(normalized, self.validation_markers):
            return "validation"
        if self.has_any_marker(normalized, self.completion_markers):
            return "done"
        if self.has_any_marker(normalized, self.implementation_markers):
            return "execution"
        return ""

    def is_plan_approval_intent(self, user_text: str) -> bool:
        normalized = user_text.strip().lower()
        return self.has_any_marker(normalized, self.plan_approval_markers)

    def is_implementation_intent(self, user_text: str) -> bool:
        normalized = user_text.strip().lower()
        return self.has_any_marker(normalized, self.implementation_markers)

    def apply_stage_transition(
        self,
        task_state: TaskState,
        desired_stage: str,
    ) -> str:
        normalized = normalize_task_stage(desired_stage)
        if not normalized:
            return ""
        try:
            task_state.transition_to(normalized)
            return ""
        except TaskTransitionError as error:
            return str(error)

    @staticmethod
    def has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)
