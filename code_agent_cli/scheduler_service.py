from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from code_agent_cli.agent import env_float, https_ssl_context
from code_agent_cli.scheduler_storage import (
    SchedulerJob,
    SchedulerRun,
    SchedulerStorage,
)


class SchedulerError(Exception):
    """Raised when a scheduler operation cannot be completed."""


class SchedulerLLMError(SchedulerError):
    """Raised when a scheduled LLM summary cannot be generated."""


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def to_iso(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise SchedulerError("datetime не должен быть пустым.")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise SchedulerError(
            "datetime должен быть ISO 8601, например 2026-06-24T12:30:00Z."
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


class SchedulerService:
    def __init__(self, db_path: Path | None = None) -> None:
        self.storage = SchedulerStorage(db_path)

    def create_reminder(self, text: str, run_at: str, *, title: str | None = None) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            raise SchedulerError("text не должен быть пустым.")

        run_at_dt = parse_datetime(run_at)
        now = utc_now()
        job = SchedulerJob(
            id=uuid4().hex,
            kind="reminder",
            title=(title or shorten_title(clean_text)).strip(),
            payload={"text": clean_text},
            schedule_type="once",
            run_at=to_iso(run_at_dt),
            interval_seconds=None,
            next_run_at=to_iso(run_at_dt),
            enabled=True,
            created_at=to_iso(now),
            updated_at=to_iso(now),
        )
        return job_to_dict(self.storage.add_job(job))

    def create_interval_summary(
        self,
        title: str,
        summary_text: str,
        interval_minutes: int,
    ) -> dict[str, Any]:
        clean_title = title.strip()
        clean_summary_text = summary_text.strip()
        if not clean_title:
            raise SchedulerError("title не должен быть пустым.")
        if not clean_summary_text:
            raise SchedulerError("summary_text не должен быть пустым.")
        if interval_minutes < 1:
            raise SchedulerError("interval_minutes должен быть положительным числом.")

        now = utc_now()
        interval_seconds = interval_minutes * 60
        next_run_at = now + timedelta(seconds=interval_seconds)
        job = SchedulerJob(
            id=uuid4().hex,
            kind="periodic_summary",
            title=clean_title,
            payload={"prompt": clean_summary_text},
            schedule_type="interval",
            run_at=None,
            interval_seconds=interval_seconds,
            next_run_at=to_iso(next_run_at),
            enabled=True,
            created_at=to_iso(now),
            updated_at=to_iso(now),
        )
        return job_to_dict(self.storage.add_job(job))

    def list_jobs(self, *, include_disabled: bool = False) -> list[dict[str, Any]]:
        return [job_to_dict(job) for job in self.storage.list_jobs(include_disabled=include_disabled)]

    def delete_job(self, job_id: str) -> dict[str, Any]:
        clean_job_id = job_id.strip()
        if not clean_job_id:
            raise SchedulerError("job_id не должен быть пустым.")
        return {"job_id": clean_job_id, "deleted": self.storage.delete_job(clean_job_id)}

    def run_due_jobs(self, *, limit: int = 20) -> dict[str, Any]:
        if limit < 1:
            raise SchedulerError("limit должен быть положительным числом.")

        started_at = utc_now()
        due_jobs = self.storage.list_due_jobs(to_iso(started_at), limit=limit)
        runs: list[dict[str, Any]] = []

        for job in due_jobs:
            runs.append(run_to_dict(self._run_job(job)))

        return {
            "checked_at": to_iso(started_at),
            "due_jobs": len(due_jobs),
            "runs": runs,
        }

    def get_summary(self, *, limit: int = 10) -> dict[str, Any]:
        if limit < 1:
            raise SchedulerError("limit должен быть положительным числом.")

        active_jobs = self.storage.list_jobs(include_disabled=False)
        runs = self.storage.list_runs(limit=limit)
        failed_runs = [run for run in runs if run.status != "success"]

        return {
            "generated_at": to_iso(utc_now()),
            "active_jobs": len(active_jobs),
            "next_runs": [job_to_dict(job) for job in active_jobs[:limit]],
            "recent_runs": [run_to_dict(run) for run in runs],
            "failed_runs": len(failed_runs),
        }

    def _run_job(self, job: SchedulerJob) -> SchedulerRun:
        started_at = utc_now()
        status = "success"
        error: str | None = None
        try:
            result = execute_job(job, started_at)
        except Exception as caught_error:
            status = "error"
            error = str(caught_error)
            result = {"message": "Job failed."}

        finished_at = utc_now()
        run = SchedulerRun(
            id=uuid4().hex,
            job_id=job.id,
            job_title=job.title,
            job_kind=job.kind,
            started_at=to_iso(started_at),
            finished_at=to_iso(finished_at),
            status=status,
            result=result,
            error=error,
        )
        self.storage.add_run(run)

        if job.schedule_type == "interval" and job.interval_seconds:
            next_run_at = finished_at + timedelta(seconds=job.interval_seconds)
            self.storage.update_job_after_run(
                job.id,
                next_run_at=to_iso(next_run_at),
                enabled=True,
                updated_at=to_iso(finished_at),
            )
        else:
            self.storage.update_job_after_run(
                job.id,
                next_run_at=None,
                enabled=False,
                updated_at=to_iso(finished_at),
            )

        return run


def execute_job(job: SchedulerJob, now: datetime) -> dict[str, Any]:
    if job.kind == "reminder":
        text = str(job.payload.get("text") or "")
        return {
            "type": "reminder",
            "title": job.title,
            "text": text,
            "message": f"Напоминание: {text}",
            "executed_at": to_iso(now),
        }

    if job.kind == "periodic_summary":
        prompt = str(job.payload.get("prompt") or job.payload.get("summary_text") or "")
        generated = generate_llm_summary(job.title, prompt, now)
        return {
            "type": "llm_summary",
            "title": job.title,
            "prompt": prompt,
            "summary": generated["content"],
            "message": generated["content"],
            "model": generated["model"],
            "usage": generated["usage"],
            "executed_at": to_iso(now),
        }

    raise SchedulerError(f"Неизвестный тип job: {job.kind}")


def generate_llm_summary(title: str, prompt: str, now: datetime) -> dict[str, Any]:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise SchedulerLLMError("prompt для LLM-сводки пуст.")

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SchedulerLLMError("DEEPSEEK_API_KEY не задан.")

    model = os.getenv("CODE_AGENT_MODEL", "deepseek-v4-flash")
    api_url = os.getenv("CODE_AGENT_API_URL", "https://api.deepseek.com/chat/completions")
    temperature = env_float("CODE_AGENT_TEMPERATURE", 0.2)
    max_tokens = int(os.getenv("CODE_AGENT_SCHEDULER_MAX_TOKENS", "700"))
    messages = [
        {
            "role": "system",
            "content": (
                "Ты фоновый агент CodeAgentCLI. По расписанию генерируй краткую, "
                "практичную сводку на русском языке. Не выдумывай факты: если "
                "данных недостаточно, явно скажи, что доступен только prompt задачи."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Название задачи: {title}\n"
                f"Время запуска UTC: {to_iso(now)}\n\n"
                "Инструкция для регулярной сводки:\n"
                f"{clean_prompt}\n\n"
                "Верни 3-7 коротких пунктов или один короткий абзац, если пунктов мало."
            ),
        },
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120, context=https_ssl_context()) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as error:
        response_text = error.read().decode("utf-8")
        raise SchedulerLLMError(f"LLM API вернул HTTP {error.code}: {response_text}") from error
    except OSError as error:
        raise SchedulerLLMError(f"LLM API недоступен: {error}") from error

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise SchedulerLLMError("LLM API вернул некорректный JSON.") from error

    choices = response_payload.get("choices") or []
    usage = response_payload.get("usage") or {}
    if not choices:
        raise SchedulerLLMError("LLM API не вернул choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise SchedulerLLMError("LLM API вернул пустую сводку.")

    return {
        "content": content.strip(),
        "model": model,
        "usage": usage if isinstance(usage, dict) else {},
    }


def job_to_dict(job: SchedulerJob) -> dict[str, Any]:
    payload = asdict(job)
    payload["enabled"] = bool(job.enabled)
    return payload


def run_to_dict(run: SchedulerRun) -> dict[str, Any]:
    return asdict(run)


def shorten_title(text: str, *, limit: int = 80) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."
