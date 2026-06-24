from __future__ import annotations

import argparse
import json
import time

from code_agent_cli.scheduler_service import SchedulerError, SchedulerService
from code_agent_cli.scheduler_storage import default_scheduler_db_file


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scheduler-runner",
        description="Run CodeAgentCLI scheduled jobs once or continuously.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running and check due jobs periodically.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds for --watch. Defaults to 60.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum due jobs to run per tick. Defaults to 20.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print scheduler summary instead of running jobs.",
    )
    args = parser.parse_args()

    if args.interval < 1:
        parser.error("--interval должен быть положительным числом.")
    if args.limit < 1:
        parser.error("--limit должен быть положительным числом.")

    runner = SchedulerRunner(SchedulerService())
    if args.summary:
        print_json(runner.summary(args.limit))
        return

    if args.watch:
        try:
            runner.watch(interval_seconds=args.interval, limit=args.limit)
        except KeyboardInterrupt:
            print_json({"status": "stopped"})
        return

    print_json(runner.run_once(args.limit))


class SchedulerRunner:
    def __init__(self, service: SchedulerService) -> None:
        self.service = service

    def run_once(self, limit: int) -> dict[str, object]:
        return {
            "database": str(default_scheduler_db_file()),
            "result": self.service.run_due_jobs(limit=limit),
        }

    def summary(self, limit: int) -> dict[str, object]:
        return {
            "database": str(default_scheduler_db_file()),
            "summary": self.service.get_summary(limit=limit),
        }

    def watch(self, *, interval_seconds: int, limit: int) -> None:
        print_json(
            {
                "status": "watching",
                "database": str(default_scheduler_db_file()),
                "interval_seconds": interval_seconds,
            }
        )
        while True:
            try:
                print_json(self.run_once(limit))
            except SchedulerError as error:
                print_json({"status": "error", "error": str(error)})
            time.sleep(interval_seconds)


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
