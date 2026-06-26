from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SchedulerJob:
    id: str
    kind: str
    title: str
    payload: dict[str, Any]
    schedule_type: str
    run_at: str | None
    interval_seconds: int | None
    next_run_at: str
    enabled: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SchedulerRun:
    id: str
    job_id: str
    job_title: str
    job_kind: str
    started_at: str
    finished_at: str
    status: str
    result: dict[str, Any]
    error: str | None


def default_scheduler_db_file() -> Path:
    configured_path = os.getenv("CODE_AGENT_SCHEDULER_DB")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path.home() / ".code-agent-cli" / "scheduler.db"


class SchedulerStorage:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or default_scheduler_db_file()).expanduser()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    schedule_type TEXT NOT NULL,
                    run_at TEXT,
                    interval_seconds INTEGER,
                    next_run_at TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_due
                    ON jobs(enabled, next_run_at);
                CREATE TABLE IF NOT EXISTS job_runs (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    job_title TEXT NOT NULL,
                    job_kind TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    error TEXT,
                    FOREIGN KEY(job_id) REFERENCES jobs(id)
                );
                CREATE INDEX IF NOT EXISTS idx_job_runs_finished
                    ON job_runs(finished_at DESC);
                """
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def add_job(self, job: SchedulerJob) -> SchedulerJob:
        self.initialize()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(
                    id, kind, title, payload_json, schedule_type, run_at,
                    interval_seconds, next_run_at, enabled, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.kind,
                    job.title,
                    json.dumps(job.payload, ensure_ascii=False),
                    job.schedule_type,
                    job.run_at,
                    job.interval_seconds,
                    job.next_run_at,
                    int(job.enabled),
                    job.created_at,
                    job.updated_at,
                ),
            )
        return job

    def list_jobs(self, *, include_disabled: bool = False) -> list[SchedulerJob]:
        self.initialize()
        query = "SELECT * FROM jobs"
        params: tuple[Any, ...] = ()
        if not include_disabled:
            query += " WHERE enabled = ?"
            params = (1,)
        query += " ORDER BY next_run_at ASC, created_at ASC"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [parse_job_row(row) for row in rows]

    def list_due_jobs(self, due_at: str, *, limit: int) -> list[SchedulerJob]:
        self.initialize()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE enabled = 1 AND next_run_at <= ?
                ORDER BY next_run_at ASC, created_at ASC
                LIMIT ?
                """,
                (due_at, limit),
            ).fetchall()
        return [parse_job_row(row) for row in rows]

    def update_job_after_run(
        self,
        job_id: str,
        *,
        next_run_at: str | None,
        enabled: bool,
        updated_at: str,
    ) -> None:
        self.initialize()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET next_run_at = COALESCE(?, next_run_at),
                    enabled = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_run_at, int(enabled), updated_at, job_id),
            )

    def delete_job(self, job_id: str) -> bool:
        self.initialize()
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cursor.rowcount > 0

    def clear(self) -> dict[str, int]:
        self.initialize()
        with self.connect() as connection:
            jobs_deleted = connection.execute("DELETE FROM jobs").rowcount
            runs_deleted = connection.execute("DELETE FROM job_runs").rowcount
        return {
            "jobs_deleted": max(jobs_deleted, 0),
            "runs_deleted": max(runs_deleted, 0),
        }

    def add_run(self, run: SchedulerRun) -> SchedulerRun:
        self.initialize()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO job_runs(
                    id, job_id, job_title, job_kind, started_at, finished_at,
                    status, result_json, error
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.job_id,
                    run.job_title,
                    run.job_kind,
                    run.started_at,
                    run.finished_at,
                    run.status,
                    json.dumps(run.result, ensure_ascii=False),
                    run.error,
                ),
            )
        return run

    def list_runs(self, *, limit: int) -> list[SchedulerRun]:
        self.initialize()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM job_runs
                ORDER BY finished_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [parse_run_row(row) for row in rows]


def parse_job_row(row: sqlite3.Row) -> SchedulerJob:
    return SchedulerJob(
        id=str(row["id"]),
        kind=str(row["kind"]),
        title=str(row["title"]),
        payload=json.loads(str(row["payload_json"])),
        schedule_type=str(row["schedule_type"]),
        run_at=row["run_at"],
        interval_seconds=row["interval_seconds"],
        next_run_at=str(row["next_run_at"]),
        enabled=bool(row["enabled"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def parse_run_row(row: sqlite3.Row) -> SchedulerRun:
    return SchedulerRun(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        job_title=str(row["job_title"]),
        job_kind=str(row["job_kind"]),
        started_at=str(row["started_at"]),
        finished_at=str(row["finished_at"]),
        status=str(row["status"]),
        result=json.loads(str(row["result_json"])),
        error=row["error"],
    )
