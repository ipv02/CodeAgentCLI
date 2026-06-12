from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent_cli.context import (
    DEFAULT_BRANCH,
    BranchState,
    normalize_facts,
    normalize_strategy,
)

HISTORY_VERSION = 3


@dataclass(frozen=True)
class HistoryState:
    strategy: str
    active_branch: str
    branches: dict[str, BranchState]


def default_history_file() -> Path:
    configured_path = os.getenv("CODE_AGENT_HISTORY_FILE")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path.home() / ".code-agent-cli" / "history.json"


class HistoryStorage:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_history_file()

    def load(self) -> HistoryState | None:
        try:
            raw_payload = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        strategy = normalize_strategy(payload.get("strategy"))
        active_branch = payload.get("active_branch")
        active_branch = active_branch if isinstance(active_branch, str) else DEFAULT_BRANCH

        branches_payload = payload.get("branches")
        if isinstance(branches_payload, dict):
            branches = load_branches(branches_payload)
            if branches:
                if active_branch not in branches:
                    active_branch = next(iter(branches))
                return HistoryState(
                    strategy=strategy,
                    active_branch=active_branch,
                    branches=branches,
                )

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None
        normalized_messages = [normalize_message(message) for message in messages]
        valid_messages = [message for message in normalized_messages if message is not None]
        if not valid_messages:
            return None

        summary = payload.get("summary")
        facts = normalize_facts(payload.get("facts"))
        if isinstance(summary, str) and summary.strip():
            facts.setdefault("legacy_summary", summary.strip())
        return HistoryState(
            strategy=strategy,
            active_branch=DEFAULT_BRANCH,
            branches={
                DEFAULT_BRANCH: BranchState(
                    messages=valid_messages,
                    facts=facts,
                    checkpoints={},
                )
            },
        )

    def save(
        self,
        strategy: str,
        active_branch: str,
        branches: dict[str, BranchState],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": HISTORY_VERSION,
            "strategy": normalize_strategy(strategy),
            "active_branch": active_branch,
            "branches": {
                name: branch.to_dict()
                for name, branch in branches.items()
            },
        }
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)


def load_branches(value: dict[str, Any]) -> dict[str, BranchState]:
    branches: dict[str, BranchState] = {}
    for name, branch_payload in value.items():
        if not isinstance(name, str) or not isinstance(branch_payload, dict):
            continue
        branch = BranchState.from_dict(branch_payload)
        normalized_messages = [
            normalize_message(message)
            for message in branch.messages
        ]
        valid_messages = [
            message
            for message in normalized_messages
            if message is not None
        ]
        if valid_messages:
            branch.messages = valid_messages
            branches[name] = branch
    return branches


def normalize_message(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None

    role = value.get("role")
    content = value.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        return None

    if role not in {"system", "user", "assistant"}:
        return None

    return {
        "role": role,
        "content": content,
    }
