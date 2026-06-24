from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from code_agent_cli.scheduler_storage import (
    SchedulerJob,
    SchedulerRun,
    SchedulerStorage,
)


class SchedulerError(Exception):
    """Raised when a scheduler operation cannot be completed."""


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
            payload={"summary_text": clean_summary_text},
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
        summary_text = str(job.payload.get("summary_text") or "")
        return {
            "type": "periodic_summary",
            "title": job.title,
            "summary": summary_text,
            "message": f"Регулярная сводка: {summary_text}",
            "executed_at": to_iso(now),
        }

    raise SchedulerError(f"Неизвестный тип job: {job.kind}")


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
