from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from code_agent_cli.local_llm import (
    DEFAULT_LOCAL_MODEL,
    LocalLLMChatService,
    LocalLLMConnectionError,
    LocalLLMError,
    LocalLLMRequestError,
    env_int,
)


DEFAULT_LLM_SERVICE_HOST = "127.0.0.1"
DEFAULT_LLM_SERVICE_PORT = 8080
DEFAULT_LLM_SERVICE_RATE_LIMIT = 30
DEFAULT_LLM_SERVICE_MAX_BODY_BYTES = 128 * 1024
DEFAULT_LLM_SERVICE_MAX_MESSAGES = 32
DEFAULT_LLM_SERVICE_MAX_MESSAGE_CHARS = 16_000


@dataclass(frozen=True)
class LLMServiceConfig:
    host: str = field(
        default_factory=lambda: os.getenv("CODE_AGENT_LLM_SERVICE_HOST", DEFAULT_LLM_SERVICE_HOST)
    )
    port: int = field(
        default_factory=lambda: env_int("CODE_AGENT_LLM_SERVICE_PORT", DEFAULT_LLM_SERVICE_PORT)
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("CODE_AGENT_LLM_SERVICE_API_KEY", "")
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: env_int(
            "CODE_AGENT_LLM_SERVICE_RATE_LIMIT",
            DEFAULT_LLM_SERVICE_RATE_LIMIT,
        )
    )
    max_body_bytes: int = field(
        default_factory=lambda: env_int(
            "CODE_AGENT_LLM_SERVICE_MAX_BODY_BYTES",
            DEFAULT_LLM_SERVICE_MAX_BODY_BYTES,
        )
    )
    max_messages: int = field(
        default_factory=lambda: env_int(
            "CODE_AGENT_LLM_SERVICE_MAX_MESSAGES",
            DEFAULT_LLM_SERVICE_MAX_MESSAGES,
        )
    )
    max_message_chars: int = field(
        default_factory=lambda: env_int(
            "CODE_AGENT_LLM_SERVICE_MAX_MESSAGE_CHARS",
            DEFAULT_LLM_SERVICE_MAX_MESSAGE_CHARS,
        )
    )

    def normalized(self) -> LLMServiceConfig:
        return LLMServiceConfig(
            host=self.host.strip() or DEFAULT_LLM_SERVICE_HOST,
            port=max(int(self.port), 1),
            api_key=self.api_key,
            rate_limit_per_minute=max(int(self.rate_limit_per_minute), 1),
            max_body_bytes=max(int(self.max_body_bytes), 1024),
            max_messages=max(int(self.max_messages), 1),
            max_message_chars=max(int(self.max_message_chars), 1),
        )


@dataclass(frozen=True)
class LLMServiceResponse:
    status: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    content_type: str = "application/json; charset=utf-8"


class RateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = max(limit_per_minute, 1)
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[int, int]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        current_window = int((now if now is not None else time.time()) // 60)
        with self._lock:
            window, count = self._windows.get(key, (current_window, 0))
            if window != current_window:
                window = current_window
                count = 0
            if count >= self.limit_per_minute:
                self._windows[key] = (window, count)
                return False
            self._windows[key] = (window, count + 1)
            return True


class LLMServiceApp:
    def __init__(self, chat: LocalLLMChatService, config: LLMServiceConfig) -> None:
        self.chat = chat
        self.config = config.normalized()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)

    def handle(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        client_id: str,
    ) -> LLMServiceResponse:
        method = method.upper()
        route = path.split("?", 1)[0].rstrip("/") or "/"
        if route in {"/", "/chat"} and method == "GET":
            return chat_page_response(self.chat, self.config)
        if route == "/health" and method == "GET":
            return self._health()

        auth_error = self._auth_error(headers)
        if auth_error is not None:
            return auth_error

        rate_key = self._rate_key(headers, client_id)
        if not self.rate_limiter.allow(rate_key):
            return error_response(
                HTTPStatus.TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                "Слишком много запросов. Повторите позже.",
            )

        if route == "/v1/models" and method == "GET":
            return self._models()
        if route == "/v1/chat" and method == "POST":
            return self._chat(body, openai_compatible=False)
        if route == "/v1/chat/completions" and method == "POST":
            return self._chat(body, openai_compatible=True)

        return error_response(HTTPStatus.NOT_FOUND, "not_found", "Маршрут не найден.")

    def _health(self) -> LLMServiceResponse:
        try:
            model_info = self.chat.model_info()
        except LocalLLMError as error:
            return LLMServiceResponse(
                status=HTTPStatus.BAD_GATEWAY,
                payload={
                    "status": "error",
                    "service": "llm-service",
                    "model": self.chat.model,
                    "ollama_url": self.chat.ollama_url,
                    "error": str(error),
                },
            )

        return LLMServiceResponse(
            status=HTTPStatus.OK,
            payload={
                "status": "ok",
                "service": "llm-service",
                "model": self.chat.model,
                "ollama_url": self.chat.ollama_url,
                "limits": self._limits(),
                "model_info": model_info,
            },
        )

    def _models(self) -> LLMServiceResponse:
        return LLMServiceResponse(
            status=HTTPStatus.OK,
            payload={
                "object": "list",
                "data": [
                    {
                        "id": self.chat.model,
                        "object": "model",
                        "owned_by": "local-ollama",
                    }
                ],
            },
        )

    def _chat(self, body: bytes, *, openai_compatible: bool) -> LLMServiceResponse:
        if len(body) > self.config.max_body_bytes:
            return error_response(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"JSON body больше лимита {self.config.max_body_bytes} bytes.",
            )

        payload, error = self._decode_json(body)
        if error is not None:
            return error

        requested_model = payload.get("model")
        if requested_model and str(requested_model) != self.chat.model:
            return error_response(
                HTTPStatus.BAD_REQUEST,
                "model_not_available",
                f"Сервис запущен с моделью {self.chat.model}.",
            )

        messages, error = self._parse_messages(payload)
        if error is not None:
            return error

        options, error = self._parse_options(payload)
        if error is not None:
            return error

        try:
            response = self.chat.generate_payload(messages, options=options)
        except LocalLLMRequestError as error:
            status = HTTPStatus.BAD_GATEWAY
            if error.status_code == HTTPStatus.NOT_FOUND:
                status = HTTPStatus.NOT_FOUND
            return error_response(status, "ollama_error", str(error))
        except LocalLLMConnectionError as error:
            return error_response(HTTPStatus.BAD_GATEWAY, "ollama_unavailable", str(error))
        except LocalLLMError as error:
            return error_response(HTTPStatus.BAD_GATEWAY, "ollama_error", str(error))

        if openai_compatible:
            return LLMServiceResponse(
                status=HTTPStatus.OK,
                payload={
                    "id": f"chatcmpl-local-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": response["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response["content"],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": response["usage"],
                    "limits": self._limits(),
                },
            )

        return LLMServiceResponse(
            status=HTTPStatus.OK,
            payload={
                "model": response["model"],
                "content": response["content"],
                "usage": response["usage"],
                "limits": self._limits(),
            },
        )

    def _auth_error(self, headers: dict[str, str]) -> LLMServiceResponse | None:
        if not self.config.api_key:
            return None
        expected = f"Bearer {self.config.api_key}"
        actual = headers.get("authorization", "")
        if not secrets.compare_digest(actual, expected):
            return error_response(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "Нужен заголовок Authorization: Bearer <token>.",
            )
        return None

    def _rate_key(self, headers: dict[str, str], client_id: str) -> str:
        authorization = headers.get("authorization", "")
        if authorization:
            return authorization
        return client_id

    def _decode_json(
        self,
        body: bytes,
    ) -> tuple[dict[str, Any], LLMServiceResponse | None]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError:
            return {}, error_response(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "JSON body должен быть UTF-8.",
            )
        except json.JSONDecodeError as error:
            return {}, error_response(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                f"Некорректный JSON: {error.msg}.",
            )
        if not isinstance(payload, dict):
            return {}, error_response(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "JSON body должен быть object.",
            )
        return payload, None

    def _parse_messages(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, str]], LLMServiceResponse | None]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return [], error_response(
                HTTPStatus.BAD_REQUEST,
                "invalid_messages",
                "Поле messages должно быть непустым массивом.",
            )
        if len(messages) > self.config.max_messages:
            return [], error_response(
                HTTPStatus.BAD_REQUEST,
                "too_many_messages",
                f"messages больше лимита {self.config.max_messages}.",
            )

        parsed: list[dict[str, str]] = []
        allowed_roles = {"system", "user", "assistant", "tool"}
        for item in messages:
            if not isinstance(item, dict):
                return [], error_response(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_messages",
                    "Каждый message должен быть object.",
                )
            role = str(item.get("role") or "")
            content = item.get("content")
            if role not in allowed_roles:
                return [], error_response(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_role",
                    f"Недопустимая роль message: {role or '<empty>'}.",
                )
            if not isinstance(content, str):
                return [], error_response(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_content",
                    "message.content должен быть строкой.",
                )
            if len(content) > self.config.max_message_chars:
                return [], error_response(
                    HTTPStatus.BAD_REQUEST,
                    "message_too_large",
                    f"message.content больше лимита {self.config.max_message_chars} символов.",
                )
            parsed.append({"role": role, "content": content})
        return parsed, None

    def _parse_options(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], LLMServiceResponse | None]:
        options_payload = payload.get("options")
        options: dict[str, Any] = {}
        if isinstance(options_payload, dict):
            for key in ("temperature", "num_predict", "num_ctx"):
                if key in options_payload:
                    options[key] = options_payload[key]

        if "temperature" in payload:
            options["temperature"] = payload["temperature"]
        if "max_tokens" in payload:
            options["num_predict"] = payload["max_tokens"]
        if "num_predict" in payload:
            options["num_predict"] = payload["num_predict"]
        if "num_ctx" in payload:
            options["num_ctx"] = payload["num_ctx"]

        if "temperature" in options:
            try:
                temperature = float(options["temperature"])
            except (TypeError, ValueError):
                return {}, error_response(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_temperature",
                    "temperature должен быть числом.",
                )
            if not 0 <= temperature <= 2:
                return {}, error_response(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_temperature",
                    "temperature должен быть в диапазоне 0-2.",
                )
            options["temperature"] = temperature

        for key, max_value in (
            ("num_predict", self.chat.num_predict),
            ("num_ctx", self.chat.num_ctx),
        ):
            if key not in options:
                continue
            try:
                value = int(options[key])
            except (TypeError, ValueError):
                return {}, error_response(
                    HTTPStatus.BAD_REQUEST,
                    f"invalid_{key}",
                    f"{key} должен быть целым числом.",
                )
            if value < 1:
                return {}, error_response(
                    HTTPStatus.BAD_REQUEST,
                    f"invalid_{key}",
                    f"{key} должен быть положительным.",
                )
            if value > max_value:
                return {}, error_response(
                    HTTPStatus.BAD_REQUEST,
                    f"{key}_limit_exceeded",
                    f"{key} больше лимита сервиса {max_value}.",
                )
            options[key] = value

        return options, None

    def _limits(self) -> dict[str, Any]:
        return {
            "rate_limit_per_minute": self.config.rate_limit_per_minute,
            "max_body_bytes": self.config.max_body_bytes,
            "max_messages": self.config.max_messages,
            "max_message_chars": self.config.max_message_chars,
            "num_ctx": self.chat.num_ctx,
            "num_predict": self.chat.num_predict,
        }


def run_llm_service(
    *,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> None:
    chat = LocalLLMChatService(model=model or os.getenv("CODE_AGENT_LOCAL_MODEL", DEFAULT_LOCAL_MODEL))
    config = LLMServiceConfig(
        host=host or os.getenv("CODE_AGENT_LLM_SERVICE_HOST", DEFAULT_LLM_SERVICE_HOST),
        port=port or env_int("CODE_AGENT_LLM_SERVICE_PORT", DEFAULT_LLM_SERVICE_PORT),
        api_key=api_key if api_key is not None else os.getenv("CODE_AGENT_LLM_SERVICE_API_KEY", ""),
    ).normalized()
    validate_service_exposure(config)
    app = LLMServiceApp(chat, config)

    handler_class = make_handler(app)
    server = ThreadingHTTPServer((config.host, config.port), handler_class)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def validate_service_exposure(config: LLMServiceConfig) -> None:
    host = config.host.lower()
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in loopback_hosts and not config.api_key:
        raise ValueError(
            "Для сетевого LLM-сервиса задайте CODE_AGENT_LLM_SERVICE_API_KEY "
            "или --llm-service-api-key."
        )


def make_handler(app: LLMServiceApp) -> type[BaseHTTPRequestHandler]:
    class LLMServiceHandler(BaseHTTPRequestHandler):
        server_version = "CodeAgentLLMService/0.1"

        def do_GET(self) -> None:
            self._handle_request()

        def do_POST(self) -> None:
            self._handle_request()

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_request(self) -> None:
            content_length = self.headers.get("Content-Length", "0")
            try:
                body_size = int(content_length)
            except ValueError:
                self._send(
                    error_response(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_content_length",
                        "Некорректный Content-Length.",
                    )
                )
                return
            if body_size > app.config.max_body_bytes:
                self._send(
                    error_response(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "request_too_large",
                        f"JSON body больше лимита {app.config.max_body_bytes} bytes.",
                    )
                )
                return

            body = self.rfile.read(body_size) if body_size else b"{}"
            client_id = self.client_address[0] if self.client_address else "unknown"
            headers = {key.lower(): value for key, value in self.headers.items()}
            response = app.handle(self.command, self.path, headers, body, client_id)
            self._send(response)

        def _send(self, response: LLMServiceResponse) -> None:
            encoded = response.body
            if encoded is None:
                encoded = json.dumps(response.payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(response.status))
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(encoded)))
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(encoded)

    return LLMServiceHandler


def error_response(
    status: HTTPStatus,
    code: str,
    message: str,
) -> LLMServiceResponse:
    return LLMServiceResponse(
        status=int(status),
        payload={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def chat_page_response(
    chat: LocalLLMChatService,
    config: LLMServiceConfig,
) -> LLMServiceResponse:
    page_config = {
        "model": chat.model,
        "authRequired": bool(config.api_key),
        "limits": {
            "rateLimitPerMinute": config.rate_limit_per_minute,
            "maxMessages": config.max_messages,
            "numCtx": chat.num_ctx,
            "numPredict": chat.num_predict,
        },
    }
    html = build_chat_page(page_config)
    return LLMServiceResponse(
        status=HTTPStatus.OK,
        payload={},
        body=html.encode("utf-8"),
        content_type="text/html; charset=utf-8",
    )


def build_chat_page(page_config: dict[str, Any]) -> str:
    config_json = json.dumps(page_config, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CodeAgentCLI LLM Chat</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #1f2328;
      --muted: #6b7280;
      --border: #d7d7d0;
      --accent: #0f766e;
      --accent-text: #ffffff;
      --error: #b91c1c;
      --user: #e8f3ff;
      --assistant: #f1f5f2;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #171918;
        --panel: #202321;
        --text: #f1f5f2;
        --muted: #a0aaa3;
        --border: #3b403d;
        --accent: #2dd4bf;
        --accent-text: #082f2a;
        --error: #fca5a5;
        --user: #16324a;
        --assistant: #29312d;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      max-width: 980px;
      margin: 0 auto;
      padding: 18px;
      gap: 14px;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 10px;
      color: var(--muted);
      font-size: 15px;
    }}
    .pill {{
      border: 1px solid color-mix(in srgb, var(--accent) 42%, var(--border));
      border-radius: 999px;
      padding: 7px 12px;
      background: color-mix(in srgb, var(--accent) 14%, var(--panel));
      color: var(--accent);
      font-weight: 800;
      box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--panel) 70%, transparent);
    }}
    main {{
      min-height: 0;
      overflow: auto;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
    }}
    .messages {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .message {{
      max-width: 82%;
      border-radius: 8px;
      padding: 10px 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--border);
    }}
    .message.user {{
      align-self: flex-end;
      background: var(--user);
    }}
    .message.assistant {{
      align-self: flex-start;
      background: var(--assistant);
    }}
    .message.error {{
      align-self: stretch;
      max-width: 100%;
      color: var(--error);
      background: transparent;
    }}
    .message.loading {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      color: var(--muted);
    }}
    .loader {{
      width: 18px;
      height: 18px;
      border-radius: 999px;
      border: 3px solid color-mix(in srgb, var(--accent) 24%, var(--border));
      border-top-color: var(--accent);
      animation: spin 0.85s linear infinite;
      flex: 0 0 auto;
    }}
    .loading-text::after {{
      content: "";
      animation: dots 1.1s steps(4, end) infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    @keyframes dots {{
      0% {{ content: ""; }}
      25% {{ content: "."; }}
      50% {{ content: ".."; }}
      75%, 100% {{ content: "..."; }}
    }}
    footer {{
      display: grid;
      gap: 10px;
    }}
    .auth {{
      display: none;
      grid-template-columns: minmax(0, 1fr);
    }}
    .auth.visible {{ display: grid; }}
    input, textarea, button {{
      font: inherit;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    input, textarea {{
      width: 100%;
      background: var(--panel);
      color: var(--text);
      padding: 10px 12px;
    }}
    textarea {{
      min-height: 76px;
      resize: vertical;
    }}
    .composer {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }}
    button {{
      min-width: 112px;
      min-height: 44px;
      padding: 0 16px;
      background: var(--accent);
      color: var(--accent-text);
      border-color: transparent;
      font-weight: 700;
      cursor: pointer;
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: wait;
    }}
    .tools {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .link-button {{
      min-width: 0;
      min-height: 0;
      padding: 0;
      border: 0;
      background: transparent;
      color: var(--muted);
      font-weight: 600;
      text-decoration: underline;
      cursor: pointer;
    }}
    @media (max-width: 680px) {{
      .shell {{ padding: 10px; }}
      header {{ align-items: flex-start; flex-direction: column; }}
      .meta {{ justify-content: flex-start; }}
      .message {{ max-width: 100%; }}
      .composer {{ grid-template-columns: 1fr; }}
      button {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>CodeAgentCLI LLM Chat</h1>
      <div class="meta">
        <span class="pill" id="model"></span>
        <span class="pill" id="limits"></span>
      </div>
    </header>
    <main id="scroll">
      <div class="messages" id="messages"></div>
    </main>
    <footer>
      <div class="auth" id="authBox">
        <input id="token" type="password" autocomplete="off" placeholder="Bearer token">
      </div>
      <form class="composer" id="form">
        <textarea id="prompt" placeholder="Введите сообщение" required></textarea>
        <button id="send" type="submit">Отправить</button>
      </form>
      <div class="tools">
        <span id="status">История хранится в этом браузере.</span>
        <button class="link-button" id="reset" type="button">Очистить</button>
      </div>
    </footer>
  </div>
  <script>
    const config = {config_json};
    const messages = [];
    const messagesEl = document.getElementById("messages");
    const scrollEl = document.getElementById("scroll");
    const formEl = document.getElementById("form");
    const promptEl = document.getElementById("prompt");
    const sendEl = document.getElementById("send");
    const tokenEl = document.getElementById("token");
    const statusEl = document.getElementById("status");
    const authBoxEl = document.getElementById("authBox");

    document.getElementById("model").textContent = config.model;
    document.getElementById("limits").textContent =
      `ctx ${{config.limits.numCtx}} · max ${{config.limits.numPredict}}`;
    if (config.authRequired) {{
      authBoxEl.classList.add("visible");
    }}

    function addMessage(role, content) {{
      const item = document.createElement("div");
      item.className = `message ${{role}}`;
      item.textContent = content;
      messagesEl.appendChild(item);
      scrollEl.scrollTop = scrollEl.scrollHeight;
      return item;
    }}

    function addLoadingMessage() {{
      const item = document.createElement("div");
      item.className = "message assistant loading";
      const spinner = document.createElement("span");
      spinner.className = "loader";
      spinner.setAttribute("aria-hidden", "true");
      const text = document.createElement("span");
      text.className = "loading-text";
      text.textContent = "Локальная модель отвечает";
      item.append(spinner, text);
      messagesEl.appendChild(item);
      scrollEl.scrollTop = scrollEl.scrollHeight;
      return item;
    }}

    function setBusy(isBusy) {{
      sendEl.disabled = isBusy;
      promptEl.disabled = isBusy;
      statusEl.textContent = isBusy ? "Локальная модель отвечает..." : "История хранится в этом браузере.";
    }}

    async function sendMessage(text) {{
      messages.push({{ role: "user", content: text }});
      addMessage("user", text);
      const loadingEl = addLoadingMessage();
      setBusy(true);
      try {{
        const headers = {{ "Content-Type": "application/json" }};
        const token = tokenEl.value.trim();
        if (token) {{
          headers.Authorization = `Bearer ${{token}}`;
        }}
        const response = await fetch("/v1/chat", {{
          method: "POST",
          headers,
          body: JSON.stringify({{ model: config.model, messages }})
        }});
        const payload = await response.json();
        if (!response.ok) {{
          const message = payload.error?.message || `HTTP ${{response.status}}`;
          throw new Error(message);
        }}
        const answer = payload.content || "";
        messages.push({{ role: "assistant", content: answer }});
        loadingEl.remove();
        addMessage("assistant", answer);
      }} catch (error) {{
        messages.pop();
        loadingEl.remove();
        addMessage("error", error.message || String(error));
      }} finally {{
        setBusy(false);
        promptEl.focus();
      }}
    }}

    formEl.addEventListener("submit", (event) => {{
      event.preventDefault();
      const text = promptEl.value.trim();
      if (!text) return;
      promptEl.value = "";
      sendMessage(text);
    }});

    document.getElementById("reset").addEventListener("click", () => {{
      messages.length = 0;
      messagesEl.replaceChildren();
      promptEl.focus();
    }});

    promptEl.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {{
        formEl.requestSubmit();
      }}
    }});

    addMessage("assistant", "Готов к чату через локальную модель.");
  </script>
</body>
</html>
"""
