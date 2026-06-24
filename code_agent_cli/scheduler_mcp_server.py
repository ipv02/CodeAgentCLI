from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from code_agent_cli.scheduler_service import SchedulerService
from code_agent_cli.scheduler_storage import default_scheduler_db_file


mcp = FastMCP(
    "CodeAgent Scheduler",
    instructions=(
        "MCP server for deferred reminders, periodic summary jobs, "
        "SQLite-backed execution history and aggregated scheduler reports."
    ),
)


def service() -> SchedulerService:
    return SchedulerService()


@mcp.tool(
    title="Scheduler health",
    description="Return scheduler storage path and service status.",
)
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "database": str(default_scheduler_db_file()),
    }


@mcp.tool(
    title="Create reminder",
    description="Create a one-time reminder scheduled at an ISO 8601 UTC datetime.",
)
def remind(text: str, run_at: str, title: str | None = None) -> dict[str, Any]:
    """Create a one-time reminder.

    Args:
        text: Reminder text.
        run_at: ISO 8601 datetime, for example 2026-06-24T12:30:00Z.
        title: Optional display title.
    """

    return service().create_reminder(text, run_at, title=title)


@mcp.tool(
    title="Create interval summary",
    description="Create a periodic summary job that runs every N minutes.",
)
def every(
    title: str,
    summary_text: str,
    interval_minutes: int,
) -> dict[str, Any]:
    """Create a periodic summary job.

    Args:
        title: Job title.
        summary_text: Text returned in each summary run.
        interval_minutes: Positive interval in minutes.
    """

    return service().create_interval_summary(title, summary_text, interval_minutes)


@mcp.tool(
    title="List scheduler jobs",
    description="List scheduled jobs, ordered by next execution time.",
)
def jobs(include_disabled: bool = False) -> dict[str, Any]:
    jobs = service().list_jobs(include_disabled=include_disabled)
    return {
        "count": len(jobs),
        "jobs": jobs,
    }


@mcp.tool(
    title="Delete scheduler job",
    description="Delete a scheduler job by id.",
)
def delete(job_id: str) -> dict[str, Any]:
    return service().delete_job(job_id)


@mcp.tool(
    title="Run due scheduler jobs",
    description="Run all scheduler jobs whose next_run_at is due and persist their results.",
)
def run_due(limit: int = 20) -> dict[str, Any]:
    return service().run_due_jobs(limit=limit)


@mcp.tool(
    title="Get scheduler summary",
    description="Return active jobs, recent job runs and aggregated failure count.",
)
def summary(limit: int = 10) -> dict[str, Any]:
    return service().get_summary(limit=limit)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
