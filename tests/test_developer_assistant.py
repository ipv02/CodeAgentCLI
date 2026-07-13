from __future__ import annotations

import subprocess
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from os import environ
from pathlib import Path
from unittest.mock import patch

from code_agent_cli.document_index import DocumentIndexError, load_project_documents
from code_agent_cli.main import (
    is_git_branch_question,
    resolve_developer_project_path,
    run_developer_help,
)
from code_agent_cli.mcp_client import MCPToolCallResult
from code_agent_cli.pipeline_service import PipelineService


class DeveloperAssistantTests(unittest.TestCase):
    def test_project_documents_include_readme_and_docs_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("# Project\n", encoding="utf-8")
            (root / "NOTES.md").write_text("not project docs\n", encoding="utf-8")
            (root / "source.py").write_text("print('skip')\n", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs" / "architecture.md").write_text("# Architecture\n", encoding="utf-8")
            (root / "project" / "docs").mkdir(parents=True)
            (root / "project" / "docs" / "api.md").write_text("# API\n", encoding="utf-8")

            documents, skipped = load_project_documents(root, max_files=10)

        self.assertEqual(
            [document.source for document in documents],
            ["README.md", "docs/architecture.md", "project/docs/api.md"],
        )
        self.assertEqual(skipped, [])

    def test_project_documents_require_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            readme.write_text("# Project\n", encoding="utf-8")

            with self.assertRaises(DocumentIndexError):
                load_project_documents(readme, max_files=10)

    def test_git_branch_tool_matches_read_only_git_command(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        payload = PipelineService().project_git_branch(str(root))

        self.assertEqual(payload["branch"], expected)
        self.assertEqual(Path(payload["repository"]), root)
        self.assertFalse(payload["detached"])

    def test_branch_questions_are_routed_to_git_context(self) -> None:
        self.assertTrue(is_git_branch_question("Какая сейчас Git-ветка?"))
        self.assertTrue(is_git_branch_question("show current branch"))
        self.assertFalse(is_git_branch_question("Как устроены модули проекта?"))

    def test_project_root_falls_back_to_editable_package_when_started_elsewhere(self) -> None:
        expected = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("code_agent_cli.main.Path.cwd", return_value=Path(directory)),
                patch.dict(environ, {}, clear=False),
            ):
                environ.pop("CODE_AGENT_PROJECT_DIR", None)
                resolved = resolve_developer_project_path(".")

        self.assertEqual(resolved, expected)

    def test_explicit_project_path_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                resolve_developer_project_path(directory),
                Path(directory).resolve(),
            )

    def test_developer_help_combines_git_and_rag_mcp_results(self) -> None:
        branch_result = MCPToolCallResult(
            content=[],
            structured_content={
                "repository": "/project",
                "branch": "feature/docs",
                "detached": False,
            },
        )
        rag_result = MCPToolCallResult(
            content=[],
            structured_content={
                "question": "Как устроен проект?",
                "mode": "enhanced",
                "grounding_status": "grounded",
                "best_similarity": 0.91,
                "generation_provider": "cloud",
                "model": "test-model",
                "answer": "CLI отделен от сервисов.",
                "sources": [{"source": "docs/architecture.md", "section": "Архитектура"}],
                "quotes": [{"source": "docs/architecture.md", "quote": "main.py отвечает за CLI"}],
            },
        )
        output = StringIO()

        with (
            patch(
                "code_agent_cli.main.call_builtin_pipeline_tool",
                side_effect=[branch_result, rag_result],
            ) as call_tool,
            patch("code_agent_cli.main.loader", side_effect=lambda _label: nullcontext()),
            redirect_stdout(output),
        ):
            run_developer_help("Как устроен проект?", Path("/project"))

        self.assertEqual(
            [call.args[0] for call in call_tool.call_args_list],
            ["project_git_branch", "rag_answer"],
        )
        rendered = output.getvalue()
        self.assertIn("feature/docs", rendered)
        self.assertIn("CLI отделен от сервисов", rendered)
        self.assertIn("docs/architecture.md", rendered)


if __name__ == "__main__":
    unittest.main()
