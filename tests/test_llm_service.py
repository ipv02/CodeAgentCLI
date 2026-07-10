from __future__ import annotations

import json
import unittest
from typing import Any

from code_agent_cli.llm_service import LLMServiceApp, LLMServiceConfig, validate_service_exposure
from code_agent_cli.local_llm import LocalLLMChatService


class FakeLocalLLM(LocalLLMChatService):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.requests: list[dict[str, Any]] = []

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/api/show":
            return {
                "details": {
                    "family": "llama",
                    "parameter_size": "3B",
                    "quantization_level": "Q4_K_M",
                },
                "model_info": {"llama.context_length": 4096},
            }
        self.requests.append(payload)
        return {
            "model": self.model,
            "message": {"role": "assistant", "content": "Сервис отвечает."},
            "eval_count": 8,
            "eval_duration": 1_000_000_000,
        }


def json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


class LLMServiceTests(unittest.TestCase):
    def make_app(self, *, api_key: str = "secret", rate_limit: int = 30) -> LLMServiceApp:
        chat = FakeLocalLLM(model="llama3.2:3b", num_ctx=4096, num_predict=512)
        config = LLMServiceConfig(
            api_key=api_key,
            rate_limit_per_minute=rate_limit,
            max_messages=3,
            max_message_chars=100,
        )
        return LLMServiceApp(chat, config)

    def test_health_reports_model_and_limits_without_chat_request(self) -> None:
        app = self.make_app()

        response = app.handle("GET", "/health", {}, b"", "127.0.0.1")

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["status"], "ok")
        self.assertEqual(response.payload["model"], "llama3.2:3b")
        self.assertEqual(response.payload["limits"]["num_ctx"], 4096)

    def test_chat_page_is_served_without_api_call(self) -> None:
        app = self.make_app(api_key="secret")

        response = app.handle("GET", "/chat", {}, b"", "127.0.0.1")

        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "text/html; charset=utf-8")
        self.assertIsNotNone(response.body)
        body = response.body.decode("utf-8") if response.body else ""
        self.assertIn("CodeAgentCLI LLM Chat", body)
        self.assertIn("/v1/chat", body)
        self.assertIn('"authRequired": true', body)
        self.assertEqual(app.chat.requests, [])

    def test_chat_requires_bearer_token_when_configured(self) -> None:
        app = self.make_app(api_key="secret")

        response = app.handle(
            "POST",
            "/v1/chat",
            {},
            json_body({"messages": [{"role": "user", "content": "hello"}]}),
            "10.0.0.2",
        )

        self.assertEqual(response.status, 401)
        self.assertEqual(response.payload["error"]["code"], "unauthorized")

    def test_chat_returns_content_and_forwards_safe_options(self) -> None:
        app = self.make_app(api_key="secret")

        response = app.handle(
            "POST",
            "/v1/chat",
            {"authorization": "Bearer secret"},
            json_body(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.0,
                    "num_ctx": 2048,
                    "max_tokens": 128,
                }
            ),
            "10.0.0.2",
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["content"], "Сервис отвечает.")
        request = app.chat.requests[0]
        self.assertEqual(request["options"]["temperature"], 0.0)
        self.assertEqual(request["options"]["num_ctx"], 2048)
        self.assertEqual(request["options"]["num_predict"], 128)

    def test_rejects_context_above_service_limit(self) -> None:
        app = self.make_app(api_key="")

        response = app.handle(
            "POST",
            "/v1/chat",
            {},
            json_body(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "num_ctx": 8192,
                }
            ),
            "127.0.0.1",
        )

        self.assertEqual(response.status, 400)
        self.assertEqual(response.payload["error"]["code"], "num_ctx_limit_exceeded")

    def test_rate_limit_is_enforced_per_client(self) -> None:
        app = self.make_app(api_key="", rate_limit=1)
        body = json_body({"messages": [{"role": "user", "content": "hello"}]})

        first = app.handle("POST", "/v1/chat", {}, body, "127.0.0.1")
        second = app.handle("POST", "/v1/chat", {}, body, "127.0.0.1")

        self.assertEqual(first.status, 200)
        self.assertEqual(second.status, 429)
        self.assertEqual(second.payload["error"]["code"], "rate_limit_exceeded")

    def test_openai_compatible_chat_completions_shape(self) -> None:
        app = self.make_app(api_key="")

        response = app.handle(
            "POST",
            "/v1/chat/completions",
            {},
            json_body({"messages": [{"role": "user", "content": "hello"}]}),
            "127.0.0.1",
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload["object"], "chat.completion")
        self.assertEqual(
            response.payload["choices"][0]["message"]["content"],
            "Сервис отвечает.",
        )

    def test_non_loopback_service_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            validate_service_exposure(LLMServiceConfig(host="0.0.0.0", api_key=""))


if __name__ == "__main__":
    unittest.main()
