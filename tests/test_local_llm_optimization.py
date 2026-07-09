from __future__ import annotations

import unittest
from typing import Any

from code_agent_cli.local_llm import LocalLLMChatService
from code_agent_cli.rag_service import build_local_rag_prompt, local_rag_generation_profile


class RecordingLocalLLM(LocalLLMChatService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        return {
            "model": self.model,
            "message": {"role": "assistant", "content": "Ответ: document_index.db"},
            "eval_count": 20,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 100,
        }


class LocalLLMOptimizationTests(unittest.TestCase):
    def test_generate_payload_sends_explicit_ollama_options_and_metrics(self) -> None:
        chat = RecordingLocalLLM(
            temperature=0.0,
            num_predict=500,
            num_ctx=4096,
        )

        response = chat.generate_payload([{"role": "user", "content": "test"}])

        options = chat.requests[0][1]["options"]
        self.assertEqual(options["temperature"], 0.0)
        self.assertEqual(options["num_predict"], 500)
        self.assertEqual(options["num_ctx"], 4096)
        self.assertEqual(response["usage"]["tokens_per_second"], 10.0)

    def test_baseline_can_keep_ollama_context_defaults(self) -> None:
        chat = RecordingLocalLLM()

        chat.generate_payload(
            [{"role": "user", "content": "test"}],
            options={"temperature": 0.2},
            include_default_options=False,
        )

        self.assertEqual(chat.requests[0][1]["options"], {"temperature": 0.2})

    def test_optimized_prompt_has_strict_evidence_boundaries(self) -> None:
        system_prompt, user_prompt = build_local_rag_prompt(
            question="Где индекс?",
            evidence="document_index.db",
            aliases="index_documents",
            profile="optimized",
        )

        self.assertIn("Используй только факты из EVIDENCE", system_prompt)
        self.assertIn("<EVIDENCE>", user_prompt)
        self.assertIn("<QUESTION>", user_prompt)
        self.assertIn("document_index.db", user_prompt)

    def test_profiles_keep_baseline_and_optimized_parameters_separate(self) -> None:
        baseline = local_rag_generation_profile("baseline")
        optimized = local_rag_generation_profile("optimized")

        self.assertIsNone(baseline.num_predict)
        self.assertIsNone(baseline.num_ctx)
        self.assertEqual(optimized.temperature, 0.0)
        self.assertEqual(optimized.num_predict, 500)
        self.assertEqual(optimized.num_ctx, 4096)


if __name__ == "__main__":
    unittest.main()
