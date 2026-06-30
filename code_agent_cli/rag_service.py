from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from code_agent_cli.agent import env_float, https_ssl_context
from code_agent_cli.document_index import DocumentIndexError, DocumentIndexService, default_document_index_db
from code_agent_cli.rag_eval import RAG_EVAL_QUESTIONS


class RAGError(Exception):
    """Raised when a RAG request cannot be completed."""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    title: str
    section: str
    strategy: str
    text: str
    score: float

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "strategy": self.strategy,
            "score": round(self.score, 4),
        }


class RAGService:
    def __init__(
        self,
        *,
        db_path: Path | None = None,
        index_service: DocumentIndexService | None = None,
    ) -> None:
        self.db_path = db_path or default_document_index_db()
        self.index_service = index_service or DocumentIndexService(db_path=self.db_path)

    def search(self, question: str, *, top_k: int = 5, strategy: str | None = None) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("question не должен быть пустым.")
        if top_k < 1 or top_k > 20:
            raise RAGError("top_k должен быть в диапазоне 1-20.")

        query_embedding = self._embed_question(clean_question)
        chunks = self._load_chunks(strategy=strategy)
        if not chunks:
            raise RAGError("Индекс пуст. Сначала выполните /mcp index-docs PATH.")

        ranked = sorted(
            (
                chunk
                for chunk in (
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        source=row["source"],
                        title=row["title"],
                        section=row["section"],
                        strategy=row["strategy"],
                        text=row["text"],
                        score=cosine_similarity(query_embedding, row["embedding"]),
                    )
                    for row in chunks
                )
            ),
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

        return {
            "question": clean_question,
            "top_k": top_k,
            "strategy": strategy or "all",
            "chunks": [render_retrieved_chunk(chunk) for chunk in ranked],
        }

    def answer(self, question: str, *, use_rag: bool = True, top_k: int = 5) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("question не должен быть пустым.")

        retrieval_payload: dict[str, Any] | None = None
        retrieved_chunks: list[dict[str, Any]] = []
        if use_rag:
            retrieval_payload = self.search(clean_question, top_k=top_k)
            retrieved_chunks = [
                chunk for chunk in retrieval_payload.get("chunks", []) if isinstance(chunk, dict)
            ]

        llm_payload = generate_rag_llm_answer(
            clean_question,
            retrieved_chunks=retrieved_chunks,
            use_rag=use_rag,
        )
        return {
            "question": clean_question,
            "mode": "rag" if use_rag else "no_rag",
            "answer": llm_payload["content"],
            "model": llm_payload["model"],
            "usage": llm_payload["usage"],
            "retrieval": retrieval_payload,
            "sources": [
                {
                    "source": chunk.get("source", ""),
                    "section": chunk.get("section", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": chunk.get("score", 0),
                }
                for chunk in retrieved_chunks
            ],
        }

    def compare(self, question: str, *, top_k: int = 5) -> dict[str, Any]:
        without_rag = self.answer(question, use_rag=False, top_k=top_k)
        with_rag = self.answer(question, use_rag=True, top_k=top_k)
        return {
            "question": question.strip(),
            "without_rag": without_rag,
            "with_rag": with_rag,
            "quality_note": (
                "Сравните полноту, конкретность и опору на sources. "
                "RAG-ответ должен использовать найденные chunks и меньше гадать."
            ),
        }

    def eval_questions(self) -> dict[str, Any]:
        return {
            "count": len(RAG_EVAL_QUESTIONS),
            "questions": RAG_EVAL_QUESTIONS,
        }

    def evaluate(self, *, top_k: int = 5, max_questions: int = 10, run_answers: bool = True) -> dict[str, Any]:
        if max_questions < 1 or max_questions > len(RAG_EVAL_QUESTIONS):
            raise RAGError(f"max_questions должен быть в диапазоне 1-{len(RAG_EVAL_QUESTIONS)}.")

        results: list[dict[str, Any]] = []
        for item in RAG_EVAL_QUESTIONS[:max_questions]:
            question = str(item["question"])
            search_payload = self.search(question, top_k=top_k)
            chunks = [chunk for chunk in search_payload.get("chunks", []) if isinstance(chunk, dict)]
            retrieved_sources = [str(chunk.get("source", "")) for chunk in chunks]
            source_hits = expected_source_hits(
                expected_sources=[str(source) for source in item.get("expected_sources", [])],
                retrieved_sources=retrieved_sources,
            )
            entry: dict[str, Any] = {
                "question": question,
                "expected": item.get("expected", ""),
                "expected_terms": item.get("expected_terms", []),
                "expected_sources": item.get("expected_sources", []),
                "retrieved_sources": retrieved_sources,
                "source_hits": source_hits,
                "retrieval": search_payload,
            }
            if run_answers:
                comparison = self.compare(question, top_k=top_k)
                entry["without_rag"] = comparison["without_rag"]["answer"]
                entry["with_rag"] = comparison["with_rag"]["answer"]
                entry["with_rag_term_hits"] = expected_term_hits(
                    comparison["with_rag"]["answer"],
                    [str(term) for term in item.get("expected_terms", [])],
                )
                entry["without_rag_term_hits"] = expected_term_hits(
                    comparison["without_rag"]["answer"],
                    [str(term) for term in item.get("expected_terms", [])],
                )
            results.append(entry)

        return {
            "questions": len(results),
            "top_k": top_k,
            "run_answers": run_answers,
            "summary": summarize_eval_results(results, run_answers=run_answers),
            "results": results,
        }

    def _embed_question(self, question: str) -> list[float]:
        try:
            return self.index_service.embed(question)
        except DocumentIndexError as error:
            raise RAGError(str(error)) from error

    def _load_chunks(self, *, strategy: str | None = None) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            raise RAGError(f"Индекс не найден: {self.db_path}. Сначала выполните /mcp index-docs PATH.")
        if strategy is not None and strategy not in {"fixed", "structural"}:
            raise RAGError("strategy должен быть fixed, structural или пустым.")

        query = (
            "SELECT chunk_id, strategy, source, title, section, text, embedding_json "
            "FROM chunks"
        )
        params: tuple[str, ...] = ()
        if strategy:
            query += " WHERE strategy = ?"
            params = (strategy,)

        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()

        chunks: list[dict[str, Any]] = []
        for row in rows:
            try:
                embedding = [float(value) for value in json.loads(row["embedding_json"])]
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise RAGError(f"Chunk {row['chunk_id']} содержит некорректный embedding.") from error
            chunks.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "strategy": str(row["strategy"]),
                    "source": str(row["source"]),
                    "title": str(row["title"]),
                    "section": str(row["section"]),
                    "text": str(row["text"]),
                    "embedding": embedding,
                }
            )
        return chunks


def generate_rag_llm_answer(
    question: str,
    *,
    retrieved_chunks: list[dict[str, Any]],
    use_rag: bool,
) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RAGError("DEEPSEEK_API_KEY не задан.")

    model = os.getenv("CODE_AGENT_MODEL", "deepseek-v4-flash")
    api_url = os.getenv("CODE_AGENT_API_URL", "https://api.deepseek.com/chat/completions")
    temperature = env_float("CODE_AGENT_TEMPERATURE", 0.2)
    max_tokens = int(os.getenv("CODE_AGENT_RAG_MAX_TOKENS", "1100"))

    if use_rag:
        context = render_rag_context(retrieved_chunks)
        system_prompt = (
            "Ты RAG-агент CodeAgentCLI. Отвечай на русском. Используй только "
            "предоставленный контекст из локального индекса. Если контекста "
            "недостаточно, явно скажи об этом. В конце добавь Sources со списком "
            "source/section/chunk_id."
        )
        user_prompt = f"Контекст:\n{context}\n\nВопрос:\n{question}"
    else:
        system_prompt = (
            "Ты CodeAgentCLI assistant. Отвечай на русском по общим знаниям модели, "
            "без доступа к локальному индексу документов."
        )
        user_prompt = question

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120, context=https_ssl_context()) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as error:
        response_text = error.read().decode("utf-8", errors="replace")
        raise RAGError(f"LLM API вернул HTTP {error.code}: {response_text}") from error
    except OSError as error:
        raise RAGError(f"LLM API недоступен: {error}") from error

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RAGError("LLM API вернул некорректный JSON.") from error

    choices = response_payload.get("choices") or []
    usage = response_payload.get("usage") or {}
    if not choices:
        raise RAGError("LLM API не вернул choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RAGError("LLM API вернул пустой answer.")

    return {
        "content": content.strip(),
        "model": model,
        "usage": usage if isinstance(usage, dict) else {},
    }


def render_rag_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "Контекст не найден."
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            (
                f"[{index}] source={chunk.get('source', '')}; "
                f"section={chunk.get('section', '')}; "
                f"chunk_id={chunk.get('chunk_id', '')}; "
                f"score={chunk.get('score', 0)}"
            )
        )
        lines.append(str(chunk.get("text", "")).strip()[:4000])
        lines.append("")
    return "\n".join(lines).strip()


def render_retrieved_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
    payload = chunk.metadata
    payload["text"] = chunk.text
    payload["preview"] = " ".join(chunk.text.split())[:300]
    return payload


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def expected_source_hits(
    *,
    expected_sources: list[str],
    retrieved_sources: list[str],
) -> dict[str, bool]:
    hits: dict[str, bool] = {}
    for expected_source in expected_sources:
        hits[expected_source] = any(
            source == expected_source or source.endswith(expected_source)
            for source in retrieved_sources
        )
    return hits


def expected_term_hits(answer: str, expected_terms: list[str]) -> dict[str, bool]:
    lowered = answer.lower()
    return {term: term.lower() in lowered for term in expected_terms}


def summarize_eval_results(results: list[dict[str, Any]], *, run_answers: bool) -> dict[str, Any]:
    total_expected_sources = 0
    matched_expected_sources = 0
    rag_term_hits = 0
    no_rag_term_hits = 0
    total_terms = 0
    for result in results:
        source_hits = result.get("source_hits")
        if isinstance(source_hits, dict):
            total_expected_sources += len(source_hits)
            matched_expected_sources += sum(1 for matched in source_hits.values() if matched)
        if run_answers:
            with_rag_hits = result.get("with_rag_term_hits")
            without_rag_hits = result.get("without_rag_term_hits")
            if isinstance(with_rag_hits, dict):
                total_terms += len(with_rag_hits)
                rag_term_hits += sum(1 for matched in with_rag_hits.values() if matched)
            if isinstance(without_rag_hits, dict):
                no_rag_term_hits += sum(1 for matched in without_rag_hits.values() if matched)

    summary: dict[str, Any] = {
        "expected_source_matches": f"{matched_expected_sources}/{total_expected_sources}",
    }
    if run_answers:
        summary["with_rag_expected_term_matches"] = f"{rag_term_hits}/{total_terms}"
        summary["without_rag_expected_term_matches"] = f"{no_rag_term_hits}/{total_terms}"
    return summary
