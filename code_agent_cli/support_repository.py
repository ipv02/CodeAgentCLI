from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


MAX_SUPPORT_DATA_BYTES = 1_000_000
USER_FIELDS = {
    "id",
    "plan",
    "account_status",
    "auth_provider",
    "mfa_enabled",
    "locale",
    "last_successful_login_at",
}
TICKET_FIELDS = {
    "id",
    "user_id",
    "status",
    "priority",
    "category",
    "subject",
    "description",
    "created_at",
    "diagnostics",
}
DIAGNOSTIC_FIELDS = {
    "error_code",
    "failed_attempts",
    "password_changed_at",
    "client",
}


class SupportDataError(ValueError):
    """Raised when support JSON is missing or invalid."""


def default_support_data_file() -> Path:
    configured = os.getenv("CODE_AGENT_SUPPORT_DATA_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path(str(files("code_agent_cli.support_data").joinpath("support.json")))


def default_support_faq_path() -> Path:
    configured = os.getenv("CODE_AGENT_SUPPORT_DOCS")
    if configured:
        return Path(configured).expanduser()
    return Path(str(files("code_agent_cli.support_data").joinpath("faq.md")))


@dataclass(frozen=True)
class SupportRepository:
    path: Path

    @classmethod
    def from_default(cls) -> "SupportRepository":
        return cls(default_support_data_file())

    def get_user(self, user_id: str) -> dict[str, Any]:
        clean_id = normalize_identifier(user_id, field="user_id")
        payload = self._load()
        users = index_records(payload["users"], record_type="user")
        user = users.get(clean_id)
        if user is None:
            raise SupportDataError(f"Пользователь не найден: {clean_id}")
        return sanitize_user(user)

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        clean_id = normalize_identifier(ticket_id, field="ticket_id")
        payload = self._load()
        tickets = index_records(payload["tickets"], record_type="ticket")
        ticket = tickets.get(clean_id)
        if ticket is None:
            raise SupportDataError(f"Тикет не найден: {clean_id}")
        return sanitize_ticket(ticket)

    def get_ticket_context(self, ticket_id: str) -> dict[str, Any]:
        ticket = self.get_ticket(ticket_id)
        user = self.get_user(str(ticket["user_id"]))
        return {
            "ticket": ticket,
            "user": user,
            "source": "support-json",
        }

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        path = self.path.expanduser()
        if not path.is_file():
            raise SupportDataError(f"JSON поддержки не найден: {path}")
        if path.stat().st_size > MAX_SUPPORT_DATA_BYTES:
            raise SupportDataError(
                f"JSON поддержки превышает лимит {MAX_SUPPORT_DATA_BYTES} байт."
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as error:
            raise SupportDataError(f"Не удалось прочитать JSON поддержки: {error}") from error
        except json.JSONDecodeError as error:
            raise SupportDataError(f"JSON поддержки некорректен: {error.msg}") from error

        if not isinstance(raw, dict):
            raise SupportDataError("JSON поддержки должен быть объектом.")
        users = validate_record_list(raw.get("users"), record_type="user")
        tickets = validate_record_list(raw.get("tickets"), record_type="ticket")
        user_ids = set(index_records(users, record_type="user"))
        for ticket in tickets:
            user_id = ticket.get("user_id")
            if not isinstance(user_id, str) or user_id not in user_ids:
                raise SupportDataError(
                    f"Тикет {ticket.get('id', '<unknown>')} ссылается на неизвестного пользователя."
                )
        return {"users": users, "tickets": tickets}


def normalize_identifier(value: str, *, field: str) -> str:
    clean = value.strip().upper()
    if not clean or len(clean) > 80:
        raise SupportDataError(f"{field} должен быть непустым идентификатором.")
    return clean


def validate_record_list(value: Any, *, record_type: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SupportDataError(f"Поле {record_type}s должно быть списком.")
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise SupportDataError(f"Каждая запись {record_type} должна быть объектом.")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise SupportDataError(f"Запись {record_type} не содержит корректный id.")
        records.append(item)
    index_records(records, record_type=record_type)
    return records


def index_records(
    records: list[dict[str, Any]],
    *,
    record_type: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = normalize_identifier(str(record["id"]), field=f"{record_type}.id")
        if identifier in result:
            raise SupportDataError(f"Повторяющийся id {record_type}: {identifier}")
        result[identifier] = record
    return result


def sanitize_user(user: dict[str, Any]) -> dict[str, Any]:
    return {key: user[key] for key in USER_FIELDS if key in user}


def sanitize_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    clean = {key: ticket[key] for key in TICKET_FIELDS if key in ticket}
    diagnostics = clean.get("diagnostics")
    if diagnostics is not None:
        if not isinstance(diagnostics, dict):
            raise SupportDataError("ticket.diagnostics должен быть объектом.")
        clean["diagnostics"] = {
            key: diagnostics[key]
            for key in DIAGNOSTIC_FIELDS
            if key in diagnostics
        }
    return clean
