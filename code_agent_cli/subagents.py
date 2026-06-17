from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from code_agent_cli.context import build_request_messages
from code_agent_cli.memory import (
    MemoryState,
    TaskState,
    TaskStateUpdate,
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


@dataclass
class TaskStateAgent:
    max_tokens: int = 300
    continue_commands: set[str] = field(
        default_factory=lambda: {"продолжай", "continue", "resume", "продолжить"}
    )
    pause_commands: set[str] = field(
        default_factory=lambda: {"пауза", "pause"}
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
    ) -> None:
        answer_lower = answer.lower()
        user_lower = user_text.lower()

        if any(token in answer_lower for token in ("провер", "test", "compileall", "валид")):
            task_state.stage = "validation"
        elif any(token in answer_lower for token in ("готово", "заверш", "done")):
            task_state.stage = "done"
        elif any(token in answer_lower for token in ("план", "шаг", "архитектур")) and not any(
            token in answer_lower for token in ("```", "func ", "class ", "def ")
        ):
            task_state.stage = "planning"
        else:
            task_state.stage = "execution"

        if not task_state.summary:
            task_state.summary = user_text.strip()
        if not task_state.current_step:
            task_state.current_step = user_text.strip()

        if task_state.stage == "planning":
            task_state.expected_action = "перейти к реализации следующего шага"
        elif task_state.stage == "execution":
            task_state.expected_action = "продолжить реализацию или уточнить следующий рабочий шаг"
        elif task_state.stage == "validation":
            task_state.expected_action = "проверить результат и подтвердить завершение"
        elif task_state.stage == "done":
            task_state.expected_action = "задача завершена"

        if any(token in user_lower for token in self.continue_commands):
            task_state.resume()

    def set_stage(self, task_state: TaskState, stage: str) -> bool:
        normalized = normalize_task_stage(stage)
        if not normalized:
            return False
        if normalized == "paused":
            task_state.pause()
        else:
            task_state.set_stage(normalized)
        return True
