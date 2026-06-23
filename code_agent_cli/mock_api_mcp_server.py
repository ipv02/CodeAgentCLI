from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from mcp.server.fastmcp import FastMCP


API_BASE_URL = "http://jsonplaceholder.typicode.com"

mcp = FastMCP(
    "CodeAgent Mock API",
    instructions="MCP server exposing tools around the JSONPlaceholder mock HTTP API.",
)


@mcp.tool(
    title="Get mock user",
    description="Fetch a user from JSONPlaceholder by numeric user id.",
)
def get_mock_user(user_id: int) -> dict[str, Any]:
    """Return one mock user by id.

    Args:
        user_id: JSONPlaceholder user id from 1 to 10.
    """

    if user_id < 1:
        raise ValueError("user_id должен быть положительным числом.")

    payload = fetch_json(f"{API_BASE_URL}/users/{user_id}")
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Mock user не найден: {user_id}")

    company = payload.get("company") if isinstance(payload.get("company"), dict) else {}
    address = payload.get("address") if isinstance(payload.get("address"), dict) else {}

    return {
        "id": payload.get("id"),
        "name": payload.get("name"),
        "username": payload.get("username"),
        "email": payload.get("email"),
        "phone": payload.get("phone"),
        "website": payload.get("website"),
        "company": company.get("name"),
        "city": address.get("city"),
        "source": f"{API_BASE_URL}/users/{user_id}",
    }


def fetch_json(url: str) -> Any:
    try:
        with urlopen(url, timeout=20) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as error:
        raise RuntimeError(f"Mock API вернул HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError(f"Mock API недоступен: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError("Mock API не ответил вовремя.") from error

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Mock API вернул некорректный JSON.") from error


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
