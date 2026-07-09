from __future__ import annotations

import statistics
import time
from typing import Any

from code_agent_cli.local_llm import LocalLLMChatService
from code_agent_cli.rag_eval import RAG_EVAL_QUESTIONS
from code_agent_cli.rag_service import (
    DEFAULT_RAG_CANDIDATE_K,
    DEFAULT_RAG_MIN_SIMILARITY,
    DEFAULT_RAG_TOP_K,
    RAGService,
    expected_term_hits,
    generate_local_rag_llm_answer,
    local_rag_generation_profile,
    render_quotes,
    render_sources,
)


def run_local_rag_optimization(
    *,
    local_model: str | None = None,
    max_questions: int = 3,
    repeats: int = 2,
) -> dict[str, Any]:
    if max_questions < 1 or max_questions > len(RAG_EVAL_QUESTIONS):
        raise ValueError(f"max_questions должен быть в диапазоне 1-{len(RAG_EVAL_QUESTIONS)}.")
    if repeats < 1 or repeats > 5:
        raise ValueError("repeats должен быть в диапазоне 1-5.")

    chat = LocalLLMChatService(model=local_model) if local_model else LocalLLMChatService()
    model_info = chat.model_info()
    rag = RAGService()
    results: list[dict[str, Any]] = []

    for item in RAG_EVAL_QUESTIONS[:max_questions]:
        question = str(item["question"])
        expected_terms = [str(term) for term in item.get("expected_terms", [])]
        retrieval_started = time.monotonic()
        retrieval = rag.search_local(
            question,
            top_k=DEFAULT_RAG_TOP_K,
            candidate_k=DEFAULT_RAG_CANDIDATE_K,
            min_similarity=DEFAULT_RAG_MIN_SIMILARITY,
            mode="enhanced",
        )
        retrieval_ms = round((time.monotonic() - retrieval_started) * 1000)
        chunks = [
            chunk for chunk in retrieval.get("chunks", []) if isinstance(chunk, dict)
        ]

        baseline = run_generation(
            question,
            chunks=chunks,
            local_model=chat.model,
            profile="baseline",
        )
        optimized_runs = [
            run_generation(
                question,
                chunks=chunks,
                local_model=chat.model,
                profile="optimized",
            )
            for _ in range(repeats)
        ]
        optimized = optimized_runs[0]
        baseline_hits = expected_term_hits(str(baseline["content"]), expected_terms)
        optimized_hits = expected_term_hits(str(optimized["content"]), expected_terms)

        results.append(
            {
                "question": question,
                "expected": str(item.get("expected", "")),
                "expected_terms": expected_terms,
                "retrieval_ms": retrieval_ms,
                "grounding_status": retrieval.get("grounding_status", ""),
                "best_similarity": retrieval.get("best_similarity", 0),
                "sources": render_sources(chunks),
                "quotes": render_quotes(question, chunks),
                "baseline": {
                    **baseline,
                    "term_hits": baseline_hits,
                    "quality_score": quality_score(baseline_hits),
                },
                "optimized": {
                    **optimized,
                    "term_hits": optimized_hits,
                    "quality_score": quality_score(optimized_hits),
                    "repeat_answers_equal": len(
                        {normalize_answer(str(run["content"])) for run in optimized_runs}
                    )
                    == 1,
                    "repeat_quality_equal": len(
                        {
                            quality_score(
                                expected_term_hits(str(run["content"]), expected_terms)
                            )
                            for run in optimized_runs
                        }
                    )
                    == 1,
                    "repeat_elapsed_ms": [run["elapsed_ms"] for run in optimized_runs],
                },
            }
        )

    return {
        "model": chat.model,
        "model_info": model_info,
        "runtime": chat.runtime_info(),
        "questions": max_questions,
        "repeats": repeats,
        "profiles": {
            name: {
                "temperature": profile.temperature,
                "num_predict": profile.num_predict,
                "num_ctx": profile.num_ctx,
            }
            for name in ("baseline", "optimized")
            for profile in [local_rag_generation_profile(name)]
        },
        "summary": summarize_results(results),
        "results": results,
    }


def run_generation(
    question: str,
    *,
    chunks: list[dict[str, Any]],
    local_model: str,
    profile: str,
) -> dict[str, Any]:
    started_at = time.monotonic()
    payload = generate_local_rag_llm_answer(
        question,
        retrieved_chunks=chunks,
        use_rag=True,
        local_model=local_model,
        generation_profile=profile,
    )
    return {
        "content": payload["content"],
        "elapsed_ms": round((time.monotonic() - started_at) * 1000),
        "usage": payload.get("usage", {}),
        "options": payload.get("generation_options", {}),
    }


def quality_score(hits: dict[str, bool]) -> float:
    if not hits:
        return 0.0
    return round(sum(1 for matched in hits.values() if matched) / len(hits), 2)


def normalize_answer(answer: str) -> str:
    return " ".join(answer.lower().split())


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_scores = [float(item["baseline"]["quality_score"]) for item in results]
    optimized_scores = [float(item["optimized"]["quality_score"]) for item in results]
    baseline_times = [int(item["baseline"]["elapsed_ms"]) for item in results]
    optimized_times = [int(item["optimized"]["elapsed_ms"]) for item in results]
    tokens_per_second = [
        float(item["optimized"]["usage"].get("tokens_per_second") or 0)
        for item in results
        if item["optimized"]["usage"].get("tokens_per_second")
    ]
    return {
        "baseline_quality": round(statistics.mean(baseline_scores), 2),
        "optimized_quality": round(statistics.mean(optimized_scores), 2),
        "baseline_avg_ms": round(statistics.mean(baseline_times)),
        "optimized_avg_ms": round(statistics.mean(optimized_times)),
        "optimized_tokens_per_second": (
            round(statistics.mean(tokens_per_second), 2) if tokens_per_second else 0
        ),
        "stable_answers": sum(
            1 for item in results if item["optimized"]["repeat_answers_equal"]
        ),
        "stable_quality": sum(
            1 for item in results if item["optimized"]["repeat_quality_equal"]
        ),
    }
