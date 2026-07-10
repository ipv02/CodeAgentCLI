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
DEFAULT_LLM_SERVICE_USERNAME = "admin"
DEFAULT_LLM_SERVICE_SESSION_TTL_SECONDS = 12 * 60 * 60
DEFAULT_LLM_SERVICE_SYSTEM_PROMPT = (
    "Ты приватный AI-ассистент по кино, актерам, режиссерам, жанрам, "
    "истории кинематографа, рекомендациям фильмов и анализу сцен. "
    "Отвечай по-русски, полезно и структурно. Если факт может быть неточным "
    "или зависит от свежих данных, явно предупреди об ограничении локальной модели."
)


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
    username: str = field(
        default_factory=lambda: os.getenv("CODE_AGENT_LLM_SERVICE_USERNAME", DEFAULT_LLM_SERVICE_USERNAME)
    )
    password: str = field(
        default_factory=lambda: os.getenv("CODE_AGENT_LLM_SERVICE_PASSWORD", "")
    )
    session_secret: str = field(
        default_factory=lambda: os.getenv("CODE_AGENT_LLM_SERVICE_SESSION_SECRET", "")
    )
    system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "CODE_AGENT_LLM_SERVICE_SYSTEM_PROMPT",
            DEFAULT_LLM_SERVICE_SYSTEM_PROMPT,
        )
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
            username=self.username.strip() or DEFAULT_LLM_SERVICE_USERNAME,
            password=self.password,
            session_secret=self.session_secret,
            system_prompt=self.system_prompt.strip() or DEFAULT_LLM_SERVICE_SYSTEM_PROMPT,
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


class SessionStore:
    def __init__(self, secret: str, ttl_seconds: int = DEFAULT_LLM_SERVICE_SESSION_TTL_SECONDS) -> None:
        self.secret = secret or secrets.token_urlsafe(32)
        self.ttl_seconds = max(ttl_seconds, 60)
        self._lock = threading.Lock()
        self._sessions: dict[str, float] = {}

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._sessions[token] = expires_at
        return token

    def valid(self, token: str) -> bool:
        if not token:
            return False
        now = time.time()
        with self._lock:
            expires_at = self._sessions.get(token)
            if expires_at is None:
                return False
            if expires_at <= now:
                self._sessions.pop(token, None)
                return False
            return True

    def revoke(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)


class LLMServiceApp:
    def __init__(self, chat: LocalLLMChatService, config: LLMServiceConfig) -> None:
        self.chat = chat
        self.config = config.normalized()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
        self.sessions = SessionStore(self.config.session_secret)

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
        if route == "/auth/login" and method == "POST":
            return self._login(body)
        if route == "/auth/logout" and method == "POST":
            return self._logout(headers)

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
        if route == "/service/status" and method == "GET":
            return self._service_status()
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

    def _login(self, body: bytes) -> LLMServiceResponse:
        if not self.config.password:
            return error_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "password_auth_disabled",
                "Логин и пароль не настроены для браузерной авторизации.",
            )
        payload, error = self._decode_json(body)
        if error is not None:
            return error
        username = str(payload.get("username") or "")
        password = str(payload.get("password") or "")
        if not (
            secrets.compare_digest(username, self.config.username)
            and secrets.compare_digest(password, self.config.password)
        ):
            return error_response(
                HTTPStatus.UNAUTHORIZED,
                "invalid_credentials",
                "Неверный логин или пароль.",
            )
        token = self.sessions.create()
        return LLMServiceResponse(
            status=HTTPStatus.OK,
            payload={"status": "ok"},
            headers={
                "Set-Cookie": (
                    f"code_agent_llm_session={token}; HttpOnly; SameSite=Lax; Path=/; "
                    f"Max-Age={DEFAULT_LLM_SERVICE_SESSION_TTL_SECONDS}"
                )
            },
        )

    def _logout(self, headers: dict[str, str]) -> LLMServiceResponse:
        token = session_token_from_headers(headers)
        if token:
            self.sessions.revoke(token)
        return LLMServiceResponse(
            status=HTTPStatus.OK,
            payload={"status": "ok"},
            headers={
                "Set-Cookie": "code_agent_llm_session=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"
            },
        )

    def _service_status(self) -> LLMServiceResponse:
        network_mode = not is_loopback_host(self.config.host)
        return LLMServiceResponse(
            status=HTTPStatus.OK,
            payload={
                "service": "private-local-llm",
                "model": self.chat.model,
                "ollama_url": self.chat.ollama_url,
                "bind_host": self.config.host,
                "port": self.config.port,
                "network_mode": network_mode,
                "localhost_only": not network_mode,
                "chat_url_hint": (
                    f"http://SERVER_IP:{self.config.port}/chat"
                    if network_mode
                    else f"http://127.0.0.1:{self.config.port}/chat"
                ),
                "auth": {
                    "login_password_enabled": bool(self.config.password),
                    "bearer_token_enabled": bool(self.config.api_key),
                },
                "domain": "cinema",
                "api_endpoints": ["/v1/chat", "/v1/chat/completions"],
                "checks": {
                    "http_api": True,
                    "browser_chat": True,
                    "auth": bool(self.config.password or self.config.api_key),
                    "rate_limit": True,
                    "max_context": True,
                    "network_access": network_mode,
                },
                "limits": self._limits(),
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
        messages = self._with_system_prompt(messages)

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
        if self._is_authorized(headers):
            return None
        if not self.config.api_key and not self.config.password:
            return None
        expected = f"Bearer {self.config.api_key}"
        actual = headers.get("authorization", "")
        if not self.config.api_key or not secrets.compare_digest(actual, expected):
            return error_response(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "Нужна авторизация: login/password session или Authorization: Bearer <token>.",
            )
        return None

    def _is_authorized(self, headers: dict[str, str]) -> bool:
        token = session_token_from_headers(headers)
        if token and self.sessions.valid(token):
            return True
        if self.config.api_key:
            expected = f"Bearer {self.config.api_key}"
            actual = headers.get("authorization", "")
            if secrets.compare_digest(actual, expected):
                return True
        return False

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

    def _with_system_prompt(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not self.config.system_prompt:
            return messages
        return [{"role": "system", "content": self.config.system_prompt}, *messages]

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
    username: str | None = None,
    password: str | None = None,
) -> None:
    chat = LocalLLMChatService(model=model or os.getenv("CODE_AGENT_LOCAL_MODEL", DEFAULT_LOCAL_MODEL))
    config = LLMServiceConfig(
        host=host or os.getenv("CODE_AGENT_LLM_SERVICE_HOST", DEFAULT_LLM_SERVICE_HOST),
        port=port or env_int("CODE_AGENT_LLM_SERVICE_PORT", DEFAULT_LLM_SERVICE_PORT),
        api_key=api_key if api_key is not None else os.getenv("CODE_AGENT_LLM_SERVICE_API_KEY", ""),
        username=username if username is not None else os.getenv("CODE_AGENT_LLM_SERVICE_USERNAME", DEFAULT_LLM_SERVICE_USERNAME),
        password=password if password is not None else os.getenv("CODE_AGENT_LLM_SERVICE_PASSWORD", ""),
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
    if not is_loopback_host(config.host) and not (config.api_key or config.password):
        raise ValueError(
            "Для сетевого LLM-сервиса задайте CODE_AGENT_LLM_SERVICE_USERNAME "
            "и CODE_AGENT_LLM_SERVICE_PASSWORD или CODE_AGENT_LLM_SERVICE_API_KEY."
        )


def is_loopback_host(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def session_token_from_headers(headers: dict[str, str]) -> str:
    cookie_header = headers.get("cookie", "")
    for part in cookie_header.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name == "code_agent_llm_session":
            return value
    return ""


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


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
    network_mode = not is_loopback_host(config.host)
    page_config = {
        "model": chat.model,
        "authRequired": bool(config.password or config.api_key),
        "loginPasswordEnabled": bool(config.password),
        "bearerTokenEnabled": bool(config.api_key),
        "bindHost": config.host,
        "port": config.port,
        "networkMode": network_mode,
        "localhostOnly": not network_mode,
        "chatUrlHint": (
            f"http://SERVER_IP:{config.port}/chat"
            if network_mode
            else f"http://127.0.0.1:{config.port}/chat"
        ),
        "domain": "cinema",
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
  <title>Приватный кино-ассистент</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #1e1e2e;
      --mantle: #181825;
      --surface: #313244;
      --surface-soft: #45475a;
      --text: #cdd6f4;
      --subtext: #a6adc8;
      --muted: #9399b2;
      --border: #45475a;
      --accent: #cba6f7;
      --accent-2: #89b4fa;
      --green: #a6e3a1;
      --yellow: #f9e2af;
      --error: #f38ba8;
      --user: #313244;
      --assistant: #1e1e2e;
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
      height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      max-width: 920px;
      margin: 0 auto;
      padding: 22px;
      gap: 16px;
    }}
    header {{
      display: grid;
      gap: 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: linear-gradient(135deg, var(--mantle), #11111b);
      padding: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: var(--subtext);
      margin: 0;
      max-width: 740px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .pill {{
      border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--border));
      border-radius: 12px;
      padding: 8px 12px;
      background: color-mix(in srgb, var(--surface) 72%, var(--accent));
      color: var(--text);
      font-weight: 800;
      box-shadow: 0 10px 28px rgba(17, 17, 27, 0.24);
    }}
    .pill strong {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .pill span {{
      display: block;
      margin-top: 2px;
    }}
    .login-panel {{
      display: grid;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--mantle);
      padding: 14px;
      gap: 10px;
    }}
    .login-panel.hidden {{
      display: none;
    }}
    .login-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }}
    .login-error {{
      color: var(--error);
      min-height: 20px;
      font-size: 13px;
    }}
    .service-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      color: var(--muted);
      font-size: 13px;
    }}
    .secondary-button {{
      min-width: 0;
      min-height: 36px;
      padding: 0 12px;
      background: transparent;
      color: var(--accent);
      border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
    }}
    main {{
      min-height: 0;
      overflow: auto;
      border: 1px solid var(--border);
      background: var(--mantle);
      border-radius: 14px;
      padding: 14px;
    }}
    .messages {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .message {{
      max-width: 82%;
      border-radius: 12px;
      padding: 10px 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--border);
    }}
    .message.user {{
      align-self: flex-end;
      background: var(--user);
      border-color: color-mix(in srgb, var(--accent-2) 36%, var(--border));
    }}
    .message.assistant {{
      align-self: flex-start;
      background: var(--assistant);
      border-color: color-mix(in srgb, var(--green) 24%, var(--border));
    }}
    .message.error {{
      align-self: stretch;
      max-width: 100%;
      color: var(--error);
      background: transparent;
    }}
    .message-meta {{
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .message-meta span {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 2px 7px;
      background: var(--surface);
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
    .composer.hidden,
    .tools.hidden {{
      display: none;
    }}
    input, textarea, button {{
      font: inherit;
      border-radius: 12px;
      border: 1px solid var(--border);
    }}
    input, textarea {{
      width: 100%;
      background: var(--mantle);
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
      color: #11111b;
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
      .meta {{ display: grid; grid-template-columns: 1fr; }}
      .login-grid {{ grid-template-columns: 1fr; }}
      .message {{ max-width: 100%; }}
      .composer {{ grid-template-columns: 1fr; }}
      button {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Приватный кино-ассистент</h1>
      <p class="subtitle">Локальная LLM как приватный сервис: чат работает через Ollama, авторизация защищает доступ, а лимиты контролируют нагрузку.</p>
      <div class="meta">
        <span class="pill"><strong>Модель</strong><span id="model"></span></span>
        <span class="pill"><strong>Контекст</strong><span id="contextLimit"></span></span>
        <span class="pill"><strong>Ответ</strong><span id="outputLimit"></span></span>
        <span class="pill"><strong>Лимит запросов</strong><span id="rateLimit"></span></span>
      </div>
    </header>
    <section class="login-panel" id="loginPanel">
      <form class="login-grid" id="loginForm">
        <input id="username" autocomplete="username" placeholder="Логин">
        <input id="password" type="password" autocomplete="current-password" placeholder="Пароль">
        <button type="submit">Войти</button>
      </form>
      <div class="login-error" id="loginError"></div>
    </section>
    <main id="scroll">
      <div class="messages" id="messages"></div>
    </main>
    <footer>
      <div class="auth" id="authBox">
        <input id="token" type="password" autocomplete="off" placeholder="Bearer token">
      </div>
      <form class="composer hidden" id="form">
        <textarea id="prompt" placeholder="Введите сообщение" required></textarea>
        <button id="send" type="submit">Отправить</button>
      </form>
      <div class="tools hidden" id="tools">
        <span id="status">История хранится в этом браузере.</span>
        <div class="service-actions">
          <button class="link-button" id="reset" type="button">Очистить</button>
          <button class="link-button" id="logout" type="button">Выйти</button>
        </div>
      </div>
    </footer>
  </div>
  <script>
    const config = {config_json};
    const messages = [];
    const messagesEl = document.getElementById("messages");
    const scrollEl = document.getElementById("scroll");
    const formEl = document.getElementById("form");
    const toolsEl = document.getElementById("tools");
    const promptEl = document.getElementById("prompt");
    const sendEl = document.getElementById("send");
    const tokenEl = document.getElementById("token");
    const statusEl = document.getElementById("status");
    const authBoxEl = document.getElementById("authBox");
    const loginPanelEl = document.getElementById("loginPanel");
    const loginFormEl = document.getElementById("loginForm");
    const loginErrorEl = document.getElementById("loginError");
    const usernameEl = document.getElementById("username");
    const passwordEl = document.getElementById("password");
    const logoutEl = document.getElementById("logout");

    document.getElementById("model").textContent = config.model;
    document.getElementById("rateLimit").textContent = `${{config.limits.rateLimitPerMinute}} req/min`;
    document.getElementById("contextLimit").textContent = `${{config.limits.numCtx}} tokens`;
    document.getElementById("outputLimit").textContent = `${{config.limits.numPredict}} tokens`;
    if (config.bearerTokenEnabled) {{
      authBoxEl.classList.add("visible");
    }}
    if (!config.loginPasswordEnabled) {{
      loginPanelEl.classList.add("hidden");
      formEl.classList.remove("hidden");
      toolsEl.classList.remove("hidden");
    }}

    function addMessage(role, content, metaItems = []) {{
      const item = document.createElement("div");
      item.className = `message ${{role}}`;
      const contentEl = document.createElement("div");
      contentEl.textContent = content;
      item.appendChild(contentEl);
      if (metaItems.length) {{
        const meta = document.createElement("div");
        meta.className = "message-meta";
        for (const value of metaItems) {{
          const part = document.createElement("span");
          part.textContent = value;
          meta.appendChild(part);
        }}
        item.appendChild(meta);
      }}
      messagesEl.appendChild(item);
      scrollEl.scrollTop = scrollEl.scrollHeight;
      return item;
    }}

    function responseMeta(payload) {{
      const usage = payload.usage || {{}};
      const limits = payload.limits || config.limits;
      const items = [
        `rate limit: ${{limits.rate_limit_per_minute ?? config.limits.rateLimitPerMinute}} req/min`,
        `max context: ${{limits.num_ctx ?? config.limits.numCtx}}`,
        `max answer: ${{limits.num_predict ?? config.limits.numPredict}}`
      ];
      if (usage.prompt_eval_count !== undefined) {{
        items.push(`prompt tokens: ${{usage.prompt_eval_count}}`);
      }}
      if (usage.eval_count !== undefined) {{
        items.push(`answer tokens: ${{usage.eval_count}}`);
      }}
      if (usage.tokens_per_second !== undefined) {{
        items.push(`${{usage.tokens_per_second}} tok/s`);
      }}
      return items;
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
        addMessage("assistant", answer, responseMeta(payload));
      }} catch (error) {{
        messages.pop();
        loadingEl.remove();
        addMessage("error", error.message || String(error));
      }} finally {{
        setBusy(false);
        promptEl.focus();
      }}
    }}

    async function login(username, password) {{
      loginErrorEl.textContent = "";
      const response = await fetch("/auth/login", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ username, password }})
      }});
      const payload = await response.json();
      if (!response.ok) {{
        throw new Error(payload.error?.message || `HTTP ${{response.status}}`);
      }}
      loginPanelEl.classList.add("hidden");
      formEl.classList.remove("hidden");
      toolsEl.classList.remove("hidden");
      addMessage("assistant", "Можно задавать вопросы.");
    }}

    formEl.addEventListener("submit", (event) => {{
      event.preventDefault();
      const text = promptEl.value.trim();
      if (!text) return;
      promptEl.value = "";
      sendMessage(text);
    }});

    loginFormEl.addEventListener("submit", (event) => {{
      event.preventDefault();
      login(usernameEl.value.trim(), passwordEl.value)
        .catch((error) => {{ loginErrorEl.textContent = error.message || String(error); }});
    }});

    document.getElementById("reset").addEventListener("click", () => {{
      messages.length = 0;
      messagesEl.replaceChildren();
      promptEl.focus();
    }});

    logoutEl.addEventListener("click", async () => {{
      await fetch("/auth/logout", {{ method: "POST" }}).catch(() => null);
      messages.length = 0;
      messagesEl.replaceChildren();
      if (config.loginPasswordEnabled) {{
        loginPanelEl.classList.remove("hidden");
        formEl.classList.add("hidden");
        toolsEl.classList.add("hidden");
      }}
    }});

    promptEl.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {{
        formEl.requestSubmit();
      }}
    }});

    if (!config.loginPasswordEnabled) {{
      addMessage("assistant", "Можно задавать вопросы.");
    }}
  </script>
</body>
</html>
"""
