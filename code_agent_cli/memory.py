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
TASK_STAGES = {
    "planning",
    "execution",
    "validation",
    "done",
    "paused",
}
ALLOWED_TASK_TRANSITIONS = {
    "planning": {"execution", "paused"},
    "execution": {"validation", "paused"},
    "validation": {"done", "paused"},
    "done": {"planning"},
    "paused": set(),
}
TASK_STAGE_DESCRIPTIONS = {
    "planning": "сначала нужно сформировать и утвердить план",
    "execution": "после утвержденного плана можно выполнять реализацию",
    "validation": "после реализации нужно проверить результат",
    "done": "завершение допустимо только после валидации",
    "paused": "пауза сохраняет предыдущий этап и возобновляется через resume",
}
TASK_STAGE_GUIDANCE = {
    "planning": (
        "Сформируй план и явно попроси пользователя подтвердить его фразой "
        '"План утверждаю, приступай". Не переходи к реализации до подтверждения.'
    ),
    "execution": (
        "Выполняй реализацию. Когда реализация готова, явно попроси следующий "
        'обязательный шаг: "Проверь реализацию" или "Запусти валидацию".'
    ),
    "validation": (
        "Проверь результат. После успешной проверки явно сообщи, что задачу можно "
        'закрыть фразой "Заверши задачу".'
    ),
    "done": "Задача закрыта. Для новой задачи нужен переход в planning.",
    "paused": 'Задача на паузе. Для продолжения попроси пользователя написать "Продолжай".',
}
TASK_STAGE_NEXT_ACTION = {
    "planning": 'Напишите: "План утверждаю, приступай".',
    "execution": 'Напишите: "Проверь реализацию" или "Запусти валидацию".',
    "validation": 'Напишите: "Заверши задачу".',
    "done": "Начните новую задачу.",
    "paused": 'Напишите: "Продолжай".',
}

MEMORY_ROUTER_SYSTEM_PROMPT = """
Ты обновляешь явную layered memory для code assistant.
Верни только JSON object. Не добавляй markdown.
""".strip()

TASK_STATE_SYSTEM_PROMPT = """
Ты обновляешь формализованное состояние задачи для code assistant.
Верни только JSON object. Не добавляй markdown.
""".strip()

TASK_STATE_UPDATE_PROMPT = """
Обнови состояние задачи на основе последнего взаимодействия.

Состояние должно быть конечным автоматом со stage:
- planning
- execution
- validation
- done
- paused

Разрешенные переходы:
- planning -> execution
- execution -> validation
- validation -> done
- любой активный этап -> paused
- paused -> только предыдущий этап через resume
- done -> planning для новой задачи

Поля:
- stage: текущий этап
- current_step: что делается прямо сейчас
- expected_action: что ожидается следующим
- summary: краткая формулировка задачи

Правила:
- paused используй только если пользователь явно поставил задачу на паузу.
- done используй только если задача действительно завершена.
- Не предлагай execution до утверждения плана пользователем.
- Не предлагай done до validation.
- Если агент объясняет план или подход, обычно это planning.
- Если агент пишет, меняет или предлагает реализовать код, обычно это execution.
- Если агент проверяет, тестирует, валидирует или просит проверить результат, обычно это validation.
- Не оставляй JSON пустым. Возвращай все 4 поля.

Текущее состояние:
{task_state}

Последнее сообщение пользователя:
{user_message}

Последний ответ ассистента:
{assistant_message}

Верни JSON строго в формате:
{{
  "stage": "planning",
  "current_step": "...",
  "expected_action": "...",
  "summary": "..."
}}
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
    task_state: "TaskState" = field(default_factory=lambda: TaskState())

    def to_dict(self) -> dict[str, Any]:
        return {
            "working": self.working,
            "long_term": self.long_term,
            "task_state": self.task_state.to_dict(),
        }

    def to_history_dict(self) -> dict[str, Any]:
        return {
            "working": self.working,
            "task_state": self.task_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MemoryState":
        if not isinstance(value, dict):
            return cls()
        return cls(
            working=normalize_memory_layer(value.get("working")),
            long_term=normalize_memory_layer(value.get("long_term")),
            task_state=TaskState.from_dict(value.get("task_state")),
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
        task_state = TaskState.from_legacy_working(working)
        return cls(working=working, long_term=long_term, task_state=task_state)

    def apply_update(self, update: "MemoryUpdate") -> None:
        merge_layer(self.working, update.working)
        merge_layer(self.long_term, update.long_term)

    def clear_working(self) -> None:
        self.working.clear()
        self.task_state.clear()

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


@dataclass(frozen=True)
class TaskStateUpdate:
    stage: str
    current_step: str
    expected_action: str
    summary: str


class TaskTransitionError(ValueError):
    def __init__(self, current_stage: str, target_stage: str) -> None:
        self.current_stage = current_stage
        self.target_stage = target_stage
        super().__init__(format_task_transition_error(current_stage, target_stage))


@dataclass
class TaskState:
    stage: str = "planning"
    current_step: str = ""
    expected_action: str = ""
    summary: str = ""
    previous_stage: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {
            "stage": self.stage,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "summary": self.summary,
        }
        if self.previous_stage:
            payload["previous_stage"] = self.previous_stage
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> "TaskState":
        if not isinstance(value, dict):
            return cls()
        stage = normalize_task_stage(value.get("stage"))
        previous_stage = normalize_task_stage(value.get("previous_stage"), allow_empty=True)
        return cls(
            stage=stage or "planning",
            current_step=normalize_text_field(value.get("current_step")),
            expected_action=normalize_text_field(value.get("expected_action")),
            summary=normalize_text_field(value.get("summary")),
            previous_stage=previous_stage,
        )

    @classmethod
    def from_legacy_working(cls, working: dict[str, str]) -> "TaskState":
        stage = normalize_task_stage(working.get("state")) or "planning"
        current_step = working.get("current_task", "")
        expected_action = working.get("plan", "")
        summary = working.get("goal", "")
        return cls(
            stage=stage,
            current_step=current_step,
            expected_action=expected_action,
            summary=summary,
        )

    def set_stage(self, stage: str) -> None:
        self.transition_to(stage)

    def transition_to(self, stage: str) -> bool:
        normalized = normalize_task_stage(stage)
        if not normalized:
            raise ValueError("task stage must be planning, execution, validation, done, or paused")
        if normalized == self.stage:
            return False
        if normalized == "paused":
            self.pause()
            return True
        if self.stage == "paused":
            raise TaskTransitionError(self.stage, normalized)
        if not can_transition_task_stage(self.stage, normalized):
            raise TaskTransitionError(self.stage, normalized)
        self.stage = normalized
        if normalized != "paused":
            self.previous_stage = ""
        return True

    def pause(self) -> None:
        if self.stage == "paused":
            return
        if not can_transition_task_stage(self.stage, "paused"):
            raise TaskTransitionError(self.stage, "paused")
        self.previous_stage = self.stage
        self.stage = "paused"

    def resume(self) -> None:
        if self.stage != "paused":
            return
        self.stage = self.previous_stage or "execution"
        self.previous_stage = ""

    def allowed_next_stages(self) -> list[str]:
        if self.stage == "paused":
            return [self.previous_stage] if self.previous_stage else []
        return sorted(ALLOWED_TASK_TRANSITIONS.get(self.stage, set()))

    def guidance(self) -> str:
        return TASK_STAGE_GUIDANCE.get(self.stage, "")

    def next_action_hint(self) -> str:
        return TASK_STAGE_NEXT_ACTION.get(self.stage, "")

    def clear(self) -> None:
        self.stage = "planning"
        self.current_step = ""
        self.expected_action = ""
        self.summary = ""
        self.previous_stage = ""

    def merge_update(self, update: TaskStateUpdate) -> None:
        self.transition_to(update.stage)
        if update.current_step:
            self.current_step = update.current_step
        if update.expected_action:
            self.expected_action = update.expected_action
        if update.summary:
            self.summary = update.summary

    @property
    def is_empty(self) -> bool:
        return (
            self.stage == "planning"
            and not self.current_step
            and not self.expected_action
            and not self.summary
            and not self.previous_stage
        )


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


def normalize_text_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_task_stage(value: Any, allow_empty: bool = False) -> str:
    normalized = normalize_text_field(value).lower()
    if not normalized and allow_empty:
        return ""
    return normalized if normalized in TASK_STAGES else ""


def can_transition_task_stage(current_stage: str, target_stage: str) -> bool:
    current = normalize_task_stage(current_stage)
    target = normalize_task_stage(target_stage)
    if not current or not target:
        return False
    if current == target:
        return True
    return target in ALLOWED_TASK_TRANSITIONS.get(current, set())


def format_task_transition_error(current_stage: str, target_stage: str) -> str:
    current = normalize_task_stage(current_stage) or current_stage
    target = normalize_task_stage(target_stage) or target_stage
    allowed = sorted(ALLOWED_TASK_TRANSITIONS.get(current, set()))
    allowed_text = ", ".join(allowed) if allowed else "нет прямых переходов"
    reason = TASK_STAGE_DESCRIPTIONS.get(target, "переход нарушает жизненный цикл задачи")
    return (
        f"Недопустимый переход задачи: {current} -> {target}. "
        f"Разрешено из {current}: {allowed_text}. {reason}."
    )


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
    task_state_message = task_state_message_block(memory.task_state)
    working_message = memory_layer_message("Working memory", memory.working)
    if long_term_message is not None:
        messages.append(long_term_message)
    if task_state_message is not None:
        messages.append(task_state_message)
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


def task_state_message_block(task_state: TaskState) -> dict[str, str] | None:
    if task_state.is_empty:
        return None
    lines = [
        "Task state:",
        f"- stage: {task_state.stage}",
    ]
    if task_state.current_step:
        lines.append(f"- current_step: {task_state.current_step}")
    if task_state.expected_action:
        lines.append(f"- expected_action: {task_state.expected_action}")
    if task_state.summary:
        lines.append(f"- summary: {task_state.summary}")
    if task_state.previous_stage:
        lines.append(f"- previous_stage: {task_state.previous_stage}")
    allowed_next = task_state.allowed_next_stages()
    if allowed_next:
        lines.append(f"- allowed_next_stages: {', '.join(allowed_next)}")
    guidance = task_state.guidance()
    if guidance:
        lines.append(f"- lifecycle_guidance: {guidance}")
    next_action = task_state.next_action_hint()
    if next_action:
        lines.append(f"- next_user_action_hint: {next_action}")
    return {
        "role": "system",
        "content": "\n".join(lines),
    }


def build_task_state_update_messages(
    current_state: TaskState,
    user_text: str,
    assistant_text: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TASK_STATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": TASK_STATE_UPDATE_PROMPT.format(
                task_state=json.dumps(current_state.to_dict(), ensure_ascii=False, indent=2),
                user_message=user_text,
                assistant_message=assistant_text,
            ),
        },
    ]


def parse_task_state_update_response(text: str) -> TaskStateUpdate:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("task state update response is not a JSON object")

    stage = normalize_task_stage(payload.get("stage"))
    if not stage:
        raise ValueError("task state update response has invalid stage")

    return TaskStateUpdate(
        stage=stage,
        current_step=normalize_text_field(payload.get("current_step")),
        expected_action=normalize_text_field(payload.get("expected_action")),
        summary=normalize_text_field(payload.get("summary")),
    )


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
