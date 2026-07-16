from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from code_agent_cli.project_file_assistant import (
    ProjectFileAssistantError,
    ProjectFileAssistantService,
    classify_file_intent,
    extract_target_path,
    is_project_file_goal,
)
from code_agent_cli.main import render_project_file_diff, render_raw_project_file_diff
from code_agent_cli.project_files import (
    ProjectFileChange,
    ProjectFileError,
    ProjectFileService,
)


class DirectProjectFilesClient:
    def __init__(self, service: ProjectFileService) -> None:
        self.service = service

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        method = getattr(self.service, tool)
        return method(**arguments)


class ProjectFilesTests(unittest.TestCase):
    def make_project(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "README.md").write_text("# Demo\n\nOld documentation.\n", encoding="utf-8")
        (root / "AGENTS.md").write_text("# Rules\n\nKeep API stable.\n", encoding="utf-8")
        package = root / "app"
        package.mkdir()
        (package / "service.py").write_text(
            "class BillingAPI:\n    pass\n",
            encoding="utf-8",
        )
        (package / "controller.py").write_text(
            "from app.service import BillingAPI\n\napi = BillingAPI()\n",
            encoding="utf-8",
        )
        return directory, root

    def test_searches_across_multiple_project_files(self) -> None:
        _directory, root = self.make_project()
        service = ProjectFileService(root)

        result = service.search_text("BillingAPI")

        self.assertEqual(result["match_count"], 3)
        self.assertEqual(
            {item["path"] for item in result["matches"]},
            {"app/service.py", "app/controller.py"},
        )

    def test_prepare_is_dry_run_and_apply_checks_sha(self) -> None:
        _directory, root = self.make_project()
        service = ProjectFileService(root)
        original = (root / "README.md").read_text(encoding="utf-8")

        prepared = service.prepare_change(
            "README.md",
            "# Demo\n\nUpdated documentation.\n",
        )

        self.assertIn("Updated documentation", prepared["diff"])
        self.assertEqual((root / "README.md").read_text(encoding="utf-8"), original)

        applied = service.apply_change(
            "README.md",
            prepared["content"],
            expected_sha256=prepared["expected_sha256"],
        )

        self.assertTrue(applied["applied"])
        self.assertIn("Updated documentation", (root / "README.md").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ProjectFileError, "изменился после чтения"):
            service.apply_change(
                "README.md",
                "stale content",
                expected_sha256=prepared["expected_sha256"],
            )

    def test_creates_new_file_atomically_and_is_idempotent(self) -> None:
        _directory, root = self.make_project()
        service = ProjectFileService(root)
        prepared = service.prepare_change("docs/adr/demo.md", "# ADR\n\nDecision.\n")

        first = service.apply_change(
            prepared["path"],
            prepared["content"],
            expected_sha256=prepared["expected_sha256"],
        )
        second = service.apply_change(
            prepared["path"],
            prepared["content"],
            expected_sha256=first["new_sha256"],
        )

        self.assertTrue(first["applied"])
        self.assertFalse(second["applied"])
        self.assertEqual((root / "docs/adr/demo.md").read_text(encoding="utf-8"), "# ADR\n\nDecision.\n")

    def test_rejects_traversal_symlink_and_protected_state(self) -> None:
        directory, root = self.make_project()
        service = ProjectFileService(root)
        outside = Path(directory.name).parent / "outside-project-file.md"
        outside.write_text("secret", encoding="utf-8")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (root / "link.md").symlink_to(outside)
        (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

        with self.assertRaises(ProjectFileError):
            service.read_file("../outside-project-file.md")
        with self.assertRaises(ProjectFileError):
            service.read_file("link.md")
        with self.assertRaises(ProjectFileError):
            service.read_file(".env")

    def test_goal_level_search_reports_multiple_files(self) -> None:
        _directory, root = self.make_project()
        client = DirectProjectFilesClient(ProjectFileService(root))
        assistant = ProjectFileAssistantService(root, client=client)

        result = assistant.run("Найди все места, где используется BillingAPI")

        self.assertEqual(result.intent, "search")
        self.assertEqual(len(result.analyzed_files), 2)
        self.assertEqual(len(result.matches), 3)

    def test_goal_level_generation_reads_three_files_and_returns_diff(self) -> None:
        _directory, root = self.make_project()
        client = DirectProjectFilesClient(ProjectFileService(root))
        prompts: list[str] = []

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return '{"content":"# Billing API usage\\n\\nGenerated from project files.\\n"}'

        assistant = ProjectFileAssistantService(
            root,
            client=client,
            content_generator=generate,
        )

        result = assistant.run(
            "Найди использования BillingAPI и создай docs/billing-api.md",
            apply=False,
        )

        self.assertEqual(result.intent, "generate")
        self.assertEqual(len(result.analyzed_files), 3)
        self.assertIn("docs/billing-api.md", result.changes[0].diff)
        self.assertFalse((root / "docs/billing-api.md").exists())
        self.assertIn("app/service.py", prompts[0])
        self.assertIn("app/controller.py", prompts[0])

        applied = assistant.apply_changes(result.changes)

        self.assertTrue(applied[0]["applied"])
        self.assertTrue((root / "docs/billing-api.md").is_file())

    def test_regular_goal_is_not_misrouted(self) -> None:
        self.assertFalse(is_project_file_goal("Объясни, что такое dependency injection"))
        self.assertTrue(
            is_project_file_goal(
                "Проверь файлы проекта и найди все использования BillingAPI"
            )
        )

    def test_documentation_fix_goal_is_an_update_of_readme(self) -> None:
        goal = "Проверь README по текущему main.py и подготовь исправления"

        intent = classify_file_intent(goal)

        self.assertEqual(intent, "update")
        self.assertEqual(extract_target_path(goal, intent=intent), "README.md")

    def test_rejects_suspiciously_truncated_existing_document(self) -> None:
        _directory, root = self.make_project()
        long_readme = "# Demo\n\n" + ("Important existing section.\n" * 120)
        (root / "README.md").write_text(long_readme, encoding="utf-8")
        client = DirectProjectFilesClient(ProjectFileService(root))
        assistant = ProjectFileAssistantService(
            root,
            client=client,
            content_generator=lambda _prompt: '{"content":"# Too short\\n"}',
        )

        with self.assertRaisesRegex(ProjectFileAssistantError, "подозрительно короткую"):
            assistant.run("Обнови README.md на основе файлов проекта")

        self.assertEqual((root / "README.md").read_text(encoding="utf-8"), long_readme)

    def test_human_diff_hides_unified_markers_for_new_file(self) -> None:
        change = ProjectFileChange(
            path="docs/demo.md",
            content="# Demo\n\n- Item\n",
            expected_sha256="",
            diff="--- /dev/null\n+++ b/docs/demo.md\n@@ -0,0 +1,3 @@\n+# Demo\n+\n+- Item\n",
        )

        rendered = render_project_file_diff(change)

        self.assertEqual(rendered, "Новый файл: docs/demo.md\n\n# Demo\n\n- Item\n")
        self.assertNotIn("@@", rendered)
        self.assertIn("+++ b/docs/demo.md", render_raw_project_file_diff(change))

    def test_human_diff_labels_existing_file_changes(self) -> None:
        change = ProjectFileChange(
            path="README.md",
            content="# Demo\n\nNew text.\n",
            expected_sha256="known-sha",
            diff=(
                "--- a/README.md\n+++ b/README.md\n@@ -1,3 +1,3 @@\n"
                " # Demo\n \n-Old text.\n+New text.\n"
            ),
        )

        rendered = render_project_file_diff(change)

        self.assertIn("Строки 1–3:", rendered)
        self.assertIn("Удалено  │ Old text.", rendered)
        self.assertIn("Добавлено │ New text.", rendered)

    def test_tty_diff_colors_gutter_without_background_fill(self) -> None:
        new_file = ProjectFileChange(
            path="docs/demo.md",
            content="# Demo\n",
            expected_sha256="",
            diff="--- /dev/null\n+++ b/docs/demo.md\n@@ -0,0 +1 @@\n+# Demo\n",
        )
        changed_file = ProjectFileChange(
            path="README.md",
            content="New\n",
            expected_sha256="known-sha",
            diff="--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-Old\n+New\n",
        )

        with patch("code_agent_cli.main.use_color", return_value=True):
            new_rendered = render_project_file_diff(new_file)
            changed_rendered = render_project_file_diff(changed_file)

        self.assertIn("\033[38;5;114m", new_rendered)
        self.assertIn("\033[38;5;114m", changed_rendered)
        self.assertIn("\033[38;5;203m", changed_rendered)
        self.assertNotIn("48;5;", new_rendered)
        self.assertNotIn("48;5;", changed_rendered)


if __name__ == "__main__":
    unittest.main()
