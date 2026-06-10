from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable


CHAT_MESSAGE_OVERHEAD_TOKENS = 4
CHAT_REPLY_OVERHEAD_TOKENS = 3
DEFAULT_CONTEXT_LIMIT_TOKENS = 64_000
DEFAULT_INPUT_PRICE_PER_1M_TOKENS = 0.28
DEFAULT_OUTPUT_PRICE_PER_1M_TOKENS = 0.42


@dataclass(frozen=True)
class ModelTokenConfig:
    context_limit: int = DEFAULT_CONTEXT_LIMIT_TOKENS
    input_price_per_1m: float = DEFAULT_INPUT_PRICE_PER_1M_TOKENS
    output_price_per_1m: float = DEFAULT_OUTPUT_PRICE_PER_1M_TOKENS


@dataclass(frozen=True)
class TokenBreakdown:
    current_request_tokens: int
    history_tokens: int
    prompt_tokens: int
    estimated_answer_tokens: int
    total_after_answer_tokens: int
    remaining_context_tokens: int
    context_limit: int
    input_cost_usd: float
    estimated_output_cost_usd: float
    estimated_total_cost_usd: float
    overflow_tokens: int = 0

    @property
    def fits_context(self) -> bool:
        return self.overflow_tokens == 0


class TokenCounter:
    def __init__(self, model: str, config: ModelTokenConfig | None = None) -> None:
        self.model = model
        self.config = config or ModelTokenConfig()
        self._encode = load_encoder(model)

    def count_text(self, text: str) -> int:
        if not text:
            return 0

        if self._encode is not None:
            return len(self._encode(text))

        return estimate_text_tokens(text)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        total = CHAT_REPLY_OVERHEAD_TOKENS
        for message in messages:
            total += CHAT_MESSAGE_OVERHEAD_TOKENS
            total += self.count_text(message.get("role", ""))
            total += self.count_text(message.get("content", ""))
        return total

    def build_breakdown(
        self,
        messages: list[dict[str, str]],
        current_request_text: str,
        estimated_answer_tokens: int = 0,
    ) -> TokenBreakdown:
        prompt_tokens = self.count_messages(messages)
        current_request_tokens = self.count_text(current_request_text)
        history_tokens = max(prompt_tokens - current_request_tokens, 0)
        total_after_answer_tokens = prompt_tokens + estimated_answer_tokens
        remaining_context_tokens = self.config.context_limit - total_after_answer_tokens
        overflow_tokens = max(-remaining_context_tokens, 0)

        input_cost = token_cost(prompt_tokens, self.config.input_price_per_1m)
        output_cost = token_cost(
            estimated_answer_tokens,
            self.config.output_price_per_1m,
        )
        return TokenBreakdown(
            current_request_tokens=current_request_tokens,
            history_tokens=history_tokens,
            prompt_tokens=prompt_tokens,
            estimated_answer_tokens=estimated_answer_tokens,
            total_after_answer_tokens=total_after_answer_tokens,
            remaining_context_tokens=remaining_context_tokens,
            context_limit=self.config.context_limit,
            input_cost_usd=input_cost,
            estimated_output_cost_usd=output_cost,
            estimated_total_cost_usd=input_cost + output_cost,
            overflow_tokens=overflow_tokens,
        )


def load_encoder(model: str) -> Callable[[str], list[int]] | None:
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return encoding.encode


def estimate_text_tokens(text: str) -> int:
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if not pieces:
        return 0

    total = 0
    for piece in pieces:
        if re.fullmatch(r"\w+", piece, flags=re.UNICODE):
            total += max(1, math.ceil(len(piece) / 4))
        else:
            total += 1

    return total


def token_cost(tokens: int, price_per_1m: float) -> float:
    return tokens * price_per_1m / 1_000_000
