from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from code_agent_cli.memory import MemoryState, memory_messages, normalize_memory_layer


SLIDING_STRATEGY = "sliding"
MEMORY_STRATEGY = "memory"
LEGACY_FACTS_STRATEGY = "facts"
BRANCHING_STRATEGY = "branching"
CONTEXT_STRATEGIES = {SLIDING_STRATEGY, MEMORY_STRATEGY, BRANCHING_STRATEGY}
DEFAULT_BRANCH = "main"

@dataclass
class BranchState:
    messages: list[dict[str, str]]
    memory: MemoryState = field(default_factory=MemoryState)
    checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "memory": self.memory.to_history_dict(),
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BranchState":
        messages = value.get("messages")
        memory = value.get("memory")
        facts = value.get("facts")
        checkpoints = value.get("checkpoints")
        return cls(
            messages=messages if isinstance(messages, list) else [],
            memory=(
                MemoryState.from_dict(memory)
                if isinstance(memory, dict)
                else MemoryState.from_legacy_facts(normalize_facts(facts))
            ),
            checkpoints=checkpoints if isinstance(checkpoints, dict) else {},
        )


def normalize_strategy(value: str | None) -> str:
    strategy = (value or MEMORY_STRATEGY).strip().lower()
    if strategy == LEGACY_FACTS_STRATEGY:
        return MEMORY_STRATEGY
    return strategy if strategy in CONTEXT_STRATEGIES else MEMORY_STRATEGY


def normalize_facts(value: Any) -> dict[str, str]:
    return normalize_memory_layer(value)


def build_request_messages(
    system_message: dict[str, str],
    memory: MemoryState,
    recent_messages: list[dict[str, str]],
    user_message: dict[str, str],
    request_text: str,
) -> list[dict[str, str]]:
    messages = [system_message]
    messages.extend(memory_messages(memory))
    messages.extend(recent_messages)
    messages.append(user_message)
    if user_message["content"] != request_text:
        messages[-1] = {
            "role": "user",
            "content": request_text,
        }
    return messages


def trim_visible_messages(
    messages: list[dict[str, str]],
    max_messages: int,
) -> list[dict[str, str]]:
    if max_messages < 1:
        return []
    return messages[-max_messages:]


def checkpoint_state(branch: BranchState) -> dict[str, Any]:
    return {
        "messages": [dict(message) for message in branch.messages],
        "memory": branch.memory.to_history_dict(),
    }


def branch_from_checkpoint(checkpoint: dict[str, Any]) -> BranchState:
    return BranchState(
        messages=[dict(message) for message in checkpoint.get("messages", [])],
        memory=(
            MemoryState.from_dict(checkpoint.get("memory"))
            if isinstance(checkpoint.get("memory"), dict)
            else MemoryState.from_legacy_facts(normalize_facts(checkpoint.get("facts")))
        ),
        checkpoints={},
    )
