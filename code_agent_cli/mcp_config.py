from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MCPConfigError(Exception):
    """Raised when MCP configuration is missing or malformed."""


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None


@dataclass(frozen=True)
class MCPConfig:
    path: Path
    servers: list[MCPServerConfig]


def default_mcp_config_file() -> Path:
    configured_path = os.getenv("CODE_AGENT_MCP_CONFIG_FILE")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path.home() / ".code-agent-cli" / "mcp.json"


def load_mcp_config(path: Path | None = None) -> MCPConfig:
    config_path = Path(path or default_mcp_config_file()).expanduser()
    try:
        raw_payload = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise MCPConfigError(f"MCP config не найден: {config_path}") from error
    except OSError as error:
        raise MCPConfigError(f"Не удалось прочитать MCP config: {error}") from error

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        raise MCPConfigError(f"MCP config содержит некорректный JSON: {error}") from error

    return parse_mcp_config(payload, config_path)


def load_mcp_config_or_empty(path: Path | None = None) -> MCPConfig:
    config_path = Path(path or default_mcp_config_file()).expanduser()
    try:
        return load_mcp_config(config_path)
    except MCPConfigError as error:
        if config_path.exists():
            raise
        return MCPConfig(path=config_path, servers=[])


def save_mcp_config(config: MCPConfig) -> Path:
    config.path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mcpServers": {
            server.name: mcp_server_payload(server)
            for server in config.servers
        }
    }
    config.path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config.path


def add_mcp_server(
    path: Path | None,
    server: MCPServerConfig,
    *,
    overwrite: bool = False,
) -> Path:
    config = load_mcp_config_or_empty(path)
    servers = [existing for existing in config.servers if existing.name != server.name]
    if len(servers) != len(config.servers) and not overwrite:
        raise MCPConfigError(f"MCP server уже существует: {server.name}")
    servers.append(server)
    return save_mcp_config(MCPConfig(path=config.path, servers=servers))


def remove_mcp_server(path: Path | None, name: str) -> Path:
    config = load_mcp_config(path)
    servers = [server for server in config.servers if server.name != name]
    if len(servers) == len(config.servers):
        raise MCPConfigError(f"MCP server не найден: {name}")
    return save_mcp_config(MCPConfig(path=config.path, servers=servers))


def clear_mcp_servers(path: Path | None) -> Path:
    config = load_mcp_config_or_empty(path)
    return save_mcp_config(MCPConfig(path=config.path, servers=[]))


def parse_mcp_config(payload: Any, path: Path) -> MCPConfig:
    if not isinstance(payload, dict):
        raise MCPConfigError("MCP config должен быть JSON object.")

    servers_payload = payload.get("mcpServers")
    if not isinstance(servers_payload, dict):
        raise MCPConfigError("MCP config должен содержать object mcpServers.")

    servers: list[MCPServerConfig] = []
    for name, server_payload in servers_payload.items():
        if not isinstance(name, str) or not name.strip():
            raise MCPConfigError("Имя MCP server должно быть непустой строкой.")
        if not isinstance(server_payload, dict):
            raise MCPConfigError(f"MCP server {name} должен быть object.")

        command = server_payload.get("command")
        if not isinstance(command, str) or not command.strip():
            raise MCPConfigError(f"MCP server {name} должен содержать command.")

        args = normalize_string_list(server_payload.get("args", []), f"{name}.args")
        env = normalize_string_dict(server_payload.get("env", {}), f"{name}.env")
        cwd = server_payload.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise MCPConfigError(f"MCP server {name}.cwd должен быть строкой.")

        servers.append(
            MCPServerConfig(
                name=name,
                command=command,
                args=args,
                env=env,
                cwd=Path(cwd).expanduser() if cwd else None,
            )
        )

    return MCPConfig(path=path, servers=servers)


def save_default_apple_mcp_config(path: Path | None = None, *, overwrite: bool = False) -> Path:
    config_path = Path(path or default_mcp_config_file()).expanduser()
    if config_path.exists() and not overwrite:
        raise MCPConfigError(f"MCP config уже существует: {config_path}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(default_apple_mcp_config_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def default_apple_mcp_config_payload() -> dict[str, Any]:
    return {
        "mcpServers": {
            "apple-mcp": {
                "command": "bunx",
                "args": [
                    "--no-cache",
                    "apple-mcp@latest",
                ],
            },
            "cupertino": {
                "command": "cupertino",
                "args": [
                    "serve",
                    "--no-reap",
                ],
            },
        }
    }


def mcp_server_payload(server: MCPServerConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command": server.command,
    }
    if server.args:
        payload["args"] = server.args
    if server.env:
        payload["env"] = server.env
    if server.cwd is not None:
        payload["cwd"] = str(server.cwd)
    return payload


def normalize_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise MCPConfigError(f"{field_name} должен быть list.")
    if not all(isinstance(item, str) for item in value):
        raise MCPConfigError(f"{field_name} должен содержать только строки.")
    return list(value)


def normalize_string_dict(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise MCPConfigError(f"{field_name} должен быть object.")

    normalized: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or not isinstance(raw_value, str):
            raise MCPConfigError(f"{field_name} должен содержать только строки.")
        normalized[key] = raw_value
    return normalized
