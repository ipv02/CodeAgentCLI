from __future__ import annotations

import json
import math
import os
import re
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


DEFAULT_RAG_TOP_K = 5
DEFAULT_RAG_CANDIDATE_K = 12
DEFAULT_RAG_MIN_SIMILARITY = 0.35
DEFAULT_RAG_QUOTE_LIMIT = 5
MAX_RAG_TOP_K = 20
RAG_RETRIEVAL_MODES = {"baseline", "filtered", "enhanced"}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    title: str
    section: str
    strategy: str
    text: str
    score: float
    similarity: float
    lexical_score: float = 0.0
    metadata_score: float = 0.0

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "strategy": self.strategy,
            "score": round(self.score, 4),
            "similarity": round(self.similarity, 4),
            "lexical_score": round(self.lexical_score, 4),
            "metadata_score": round(self.metadata_score, 4),
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

    def search(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_RAG_TOP_K,
        candidate_k: int = DEFAULT_RAG_CANDIDATE_K,
        min_similarity: float = DEFAULT_RAG_MIN_SIMILARITY,
        strategy: str | None = None,
        mode: str = "enhanced",
    ) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("question не должен быть пустым.")
        if top_k < 1 or top_k > MAX_RAG_TOP_K:
            raise RAGError("top_k должен быть в диапазоне 1-20.")
        if min_similarity < -1 or min_similarity > 1:
            raise RAGError("min_similarity должен быть в диапазоне -1..1.")
        if mode not in RAG_RETRIEVAL_MODES:
            raise RAGError("mode должен быть baseline, filtered или enhanced.")
        if mode != "baseline" and (candidate_k < top_k or candidate_k > MAX_RAG_TOP_K):
            raise RAGError("candidate_k должен быть в диапазоне top_k-20.")

        effective_candidate_k = top_k if mode == "baseline" else candidate_k
        rewritten_question = rewrite_query(clean_question) if mode == "enhanced" else clean_question
        query_embedding = self._embed_question(rewritten_question)
        chunks = self._load_chunks(strategy=strategy)
        if not chunks:
            raise RAGError("Индекс пуст. Сначала выполните /mcp index-docs PATH.")

        scored_chunks: list[RetrievedChunk] = []
        for row in chunks:
            similarity = cosine_similarity(query_embedding, row["embedding"])
            scored_chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    title=row["title"],
                    section=row["section"],
                    strategy=row["strategy"],
                    text=row["text"],
                    score=similarity,
                    similarity=similarity,
                )
            )
        candidates = sorted(scored_chunks, key=lambda item: item.score, reverse=True)[:effective_candidate_k]

        if mode == "baseline":
            ranked = candidates[:top_k]
            candidates_after_filter = len(ranked)
            filtered_out = 0
        else:
            reranked = rerank_chunks(clean_question, candidates)
            filtered = [chunk for chunk in reranked if chunk.similarity >= min_similarity]
            candidates_after_filter = len(filtered)
            filtered_out = len(candidates) - len(filtered)
            ranked = filtered[:top_k]

        return {
            "question": clean_question,
            "rewritten_question": rewritten_question if rewritten_question != clean_question else "",
            "mode": mode,
            "top_k": top_k,
            "candidate_k": effective_candidate_k,
            "top_k_before_filter": len(candidates),
            "candidates_after_filter": candidates_after_filter,
            "top_k_after_filter": len(ranked),
            "min_similarity": None if mode == "baseline" else min_similarity,
            "filtered_out": filtered_out,
            "best_similarity": round(ranked[0].similarity, 4) if ranked else 0.0,
            "grounding_status": "grounded" if ranked else "weak_context",
            "strategy": strategy or "all",
            "chunks": [render_retrieved_chunk(clean_question, chunk) for chunk in ranked],
        }

    def answer(
        self,
        question: str,
        *,
        use_rag: bool = True,
        top_k: int = DEFAULT_RAG_TOP_K,
        candidate_k: int = DEFAULT_RAG_CANDIDATE_K,
        min_similarity: float = DEFAULT_RAG_MIN_SIMILARITY,
        mode: str = "enhanced",
    ) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("question не должен быть пустым.")

        retrieval_payload: dict[str, Any] | None = None
        retrieved_chunks: list[dict[str, Any]] = []
        grounding_status = "no_rag"
        best_similarity = 0.0
        if use_rag:
            retrieval_payload = self.search(
                clean_question,
                top_k=top_k,
                candidate_k=candidate_k,
                min_similarity=min_similarity,
                mode=mode,
            )
            retrieved_chunks = [
                chunk for chunk in retrieval_payload.get("chunks", []) if isinstance(chunk, dict)
            ]
            best_similarity = float(retrieval_payload.get("best_similarity") or 0.0)
            grounding_status = "grounded" if retrieved_chunks and best_similarity >= min_similarity else "weak_context"

        sources = render_sources(retrieved_chunks)
        quotes = render_quotes(clean_question, retrieved_chunks)

        if use_rag and grounding_status == "weak_context":
            answer = weak_context_answer()
            llm_payload = {
                "content": append_grounding_sections(answer, sources=sources, quotes=quotes),
                "raw_content": answer,
                "model": "local-grounding-policy",
                "usage": {},
            }
        else:
            llm_payload = generate_rag_llm_answer(
                clean_question,
                retrieved_chunks=retrieved_chunks,
                use_rag=use_rag,
            )
            llm_payload["raw_content"] = llm_payload["content"]
            if use_rag:
                llm_payload = {
                    **llm_payload,
                    "content": append_grounding_sections(
                        llm_payload["content"],
                        sources=sources,
                        quotes=quotes,
                    ),
                }

        return {
            "question": clean_question,
            "mode": mode if use_rag else "no_rag",
            "grounding_status": grounding_status,
            "best_similarity": round(best_similarity, 4),
            "answer": llm_payload["content"],
            "raw_answer": llm_payload.get("raw_content", llm_payload["content"]),
            "model": llm_payload["model"],
            "usage": llm_payload["usage"],
            "retrieval": retrieval_payload,
            "sources": sources,
            "quotes": quotes,
        }

    def compare(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_RAG_TOP_K,
        candidate_k: int = DEFAULT_RAG_CANDIDATE_K,
        min_similarity: float = DEFAULT_RAG_MIN_SIMILARITY,
    ) -> dict[str, Any]:
        without_rag = self.answer(question, use_rag=False, top_k=top_k)
        baseline_rag = self.answer(
            question,
            use_rag=True,
            top_k=top_k,
            candidate_k=top_k,
            min_similarity=min_similarity,
            mode="baseline",
        )
        with_rag = self.answer(
            question,
            use_rag=True,
            top_k=top_k,
            candidate_k=candidate_k,
            min_similarity=min_similarity,
            mode="enhanced",
        )
        return {
            "question": question.strip(),
            "retrieval_modes": {
                "baseline": baseline_rag.get("retrieval"),
                "enhanced": with_rag.get("retrieval"),
            },
            "without_rag": without_rag,
            "baseline_rag": baseline_rag,
            "with_rag": with_rag,
            "quality_note": (
                "Сравните baseline retrieval с enhanced retrieval: фильтр должен отсечь "
                "слабые chunks, а heuristic rerank поднять совпадения по терминам, title и section."
            ),
        }

    def eval_questions(self) -> dict[str, Any]:
        return {
            "count": len(RAG_EVAL_QUESTIONS),
            "questions": RAG_EVAL_QUESTIONS,
        }

    def evaluate(
        self,
        *,
        top_k: int = DEFAULT_RAG_TOP_K,
        candidate_k: int = DEFAULT_RAG_CANDIDATE_K,
        min_similarity: float = DEFAULT_RAG_MIN_SIMILARITY,
        max_questions: int = 10,
        run_answers: bool = True,
    ) -> dict[str, Any]:
        if max_questions < 1 or max_questions > len(RAG_EVAL_QUESTIONS):
            raise RAGError(f"max_questions должен быть в диапазоне 1-{len(RAG_EVAL_QUESTIONS)}.")

        results: list[dict[str, Any]] = []
        for item in RAG_EVAL_QUESTIONS[:max_questions]:
            question = str(item["question"])
            baseline_search = self.search(question, top_k=top_k, candidate_k=top_k, mode="baseline")
            enhanced_search = self.search(
                question,
                top_k=top_k,
                candidate_k=candidate_k,
                min_similarity=min_similarity,
                mode="enhanced",
            )
            baseline_chunks = [chunk for chunk in baseline_search.get("chunks", []) if isinstance(chunk, dict)]
            enhanced_chunks = [chunk for chunk in enhanced_search.get("chunks", []) if isinstance(chunk, dict)]
            baseline_sources = [str(chunk.get("source", "")) for chunk in baseline_chunks]
            enhanced_sources = [str(chunk.get("source", "")) for chunk in enhanced_chunks]
            source_hits = expected_source_hits(
                expected_sources=[str(source) for source in item.get("expected_sources", [])],
                retrieved_sources=enhanced_sources,
            )
            baseline_source_hits = expected_source_hits(
                expected_sources=[str(source) for source in item.get("expected_sources", [])],
                retrieved_sources=baseline_sources,
            )
            entry: dict[str, Any] = {
                "question": question,
                "expected": item.get("expected", ""),
                "expected_terms": item.get("expected_terms", []),
                "expected_sources": item.get("expected_sources", []),
                "retrieved_sources": enhanced_sources,
                "baseline_retrieved_sources": baseline_sources,
                "source_hits": source_hits,
                "baseline_source_hits": baseline_source_hits,
                "retrieval": enhanced_search,
                "baseline_retrieval": baseline_search,
            }
            if run_answers:
                comparison = self.compare(
                    question,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    min_similarity=min_similarity,
                )
                without_rag_answer = str(comparison["without_rag"].get("raw_answer", comparison["without_rag"]["answer"]))
                baseline_rag_answer = str(comparison["baseline_rag"].get("raw_answer", comparison["baseline_rag"]["answer"]))
                with_rag_answer = str(comparison["with_rag"].get("raw_answer", comparison["with_rag"]["answer"]))
                entry["without_rag"] = without_rag_answer
                entry["baseline_rag"] = baseline_rag_answer
                entry["with_rag"] = with_rag_answer
                entry["with_rag_has_sources"] = bool(comparison["with_rag"].get("sources"))
                entry["with_rag_has_quotes"] = bool(comparison["with_rag"].get("quotes"))
                entry["with_rag_grounding_status"] = comparison["with_rag"].get("grounding_status", "")
                entry["with_rag_quote_term_hits"] = expected_term_hits(
                    render_quote_text(comparison["with_rag"].get("quotes")),
                    [str(term) for term in item.get("expected_terms", [])],
                )
                entry["with_rag_answer_quote_alignment"] = answer_quote_alignment(
                    with_rag_answer,
                    render_quote_text(comparison["with_rag"].get("quotes")),
                )
                entry["with_rag_term_hits"] = expected_term_hits(
                    with_rag_answer,
                    [str(term) for term in item.get("expected_terms", [])],
                )
                entry["baseline_rag_term_hits"] = expected_term_hits(
                    baseline_rag_answer,
                    [str(term) for term in item.get("expected_terms", [])],
                )
                entry["without_rag_term_hits"] = expected_term_hits(
                    without_rag_answer,
                    [str(term) for term in item.get("expected_terms", [])],
                )
            results.append(entry)

        return {
            "questions": len(results),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "min_similarity": min_similarity,
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
            "предоставленный контекст из локального индекса. Обязательно верни "
            "структуру: Ответ, Sources, Quotes. В Sources перечисли "
            "source/section/chunk_id. В Quotes приведи короткие фрагменты только "
            "из найденных чанков. Если контекст слабый или недостаточный, ответь "
            "\"Не знаю\" и попроси уточнить вопрос. Не добавляй факты без опоры "
            "на контекст."
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
    last_empty_reason = "LLM API вернул пустой answer."
    for attempt in range(2):
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
            last_empty_reason = "LLM API не вернул choices."
            if attempt == 0:
                continue
            raise RAGError(last_empty_reason)
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            last_empty_reason = "LLM API вернул пустой answer."
            if attempt == 0:
                continue
            raise RAGError(last_empty_reason)

        return {
            "content": content.strip(),
            "model": model,
            "usage": usage if isinstance(usage, dict) else {},
        }

    raise RAGError(last_empty_reason)


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
        quote = str(chunk.get("quote", "")).strip()
        if quote:
            lines.append(f"quote={quote}")
        lines.append(str(chunk.get("text", "")).strip()[:4000])
        lines.append("")
    return "\n".join(lines).strip()


def render_retrieved_chunk(question: str, chunk: RetrievedChunk) -> dict[str, Any]:
    payload = chunk.metadata
    payload["text"] = chunk.text
    payload["quote"] = extract_quote(question, chunk.text)
    payload["preview"] = " ".join(chunk.text.split())[:300]
    return payload


def render_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": chunk.get("source", ""),
            "section": chunk.get("section", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "score": chunk.get("score", 0),
            "similarity": chunk.get("similarity", 0),
        }
        for chunk in chunks
    ]


def render_quotes(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quotes: list[dict[str, Any]] = []
    for chunk in chunks[:DEFAULT_RAG_QUOTE_LIMIT]:
        quote = extract_quote(question, str(chunk.get("text", "")))
        if not quote:
            continue
        quotes.append(
            {
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "quote": quote,
                "score": chunk.get("score", 0),
                "similarity": chunk.get("similarity", 0),
            }
        )
    return quotes


def extract_quote(question: str, text: str, *, max_chars: int = 360) -> str:
    clean_text = " ".join(text.split())
    if not clean_text:
        return ""
    terms = tokenize_for_rag(question)
    candidates = split_quote_candidates(clean_text)
    if not candidates:
        return clean_text[:max_chars].strip()
    best = max(candidates, key=lambda value: overlap_score(terms, tokenize_for_rag(value)))
    if not best.strip():
        best = candidates[0]
    if len(best) <= max_chars:
        return best.strip()
    return best[: max_chars - 1].rstrip() + "…"


def split_quote_candidates(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    if sentences:
        return sentences
    parts = [part.strip() for part in re.split(r"\n{2,}|;\s+", text) if part.strip()]
    return parts or [text.strip()]


def weak_context_answer() -> str:
    return (
        "Не знаю: в локальном индексе не найден достаточно релевантный контекст. "
        "Уточните вопрос или переиндексируйте документы, если нужная информация должна быть в проекте."
    )


def append_grounding_sections(
    answer: str,
    *,
    sources: list[dict[str, Any]],
    quotes: list[dict[str, Any]],
) -> str:
    content = answer.strip()
    sections: list[str] = [content]
    source_lines = [
        f"- {source.get('source', '')} / {source.get('section', '')} / {source.get('chunk_id', '')}"
        for source in sources
    ]
    sections.append("Verified Sources:\n" + ("\n".join(source_lines) if source_lines else "- нет релевантных источников"))
    quote_lines = [
        (
            f"- {quote.get('source', '')} / {quote.get('section', '')} / "
            f"{quote.get('chunk_id', '')}: \"{quote.get('quote', '')}\""
        )
        for quote in quotes
    ]
    sections.append("Verified Quotes:\n" + ("\n".join(quote_lines) if quote_lines else "- нет релевантных цитат"))
    return "\n\n".join(section for section in sections if section)


def render_quote_text(value: Any) -> str:
    quotes = value if isinstance(value, list) else []
    return "\n".join(str(quote.get("quote", "")) for quote in quotes if isinstance(quote, dict))


def answer_quote_alignment(answer: str, quotes_text: str) -> dict[str, Any]:
    answer_terms = tokenize_for_rag(answer)
    quote_terms = tokenize_for_rag(quotes_text)
    score = overlap_score(answer_terms, quote_terms)
    return {
        "score": round(score, 4),
        "aligned": bool(answer_terms and quote_terms and score >= 0.25),
    }


def rerank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    question_terms = tokenize_for_rag(question)
    reranked: list[RetrievedChunk] = []
    for chunk in chunks:
        text_terms = tokenize_for_rag(chunk.text)
        metadata_terms = tokenize_for_rag(" ".join([chunk.title, chunk.section, chunk.source]))
        lexical_score = overlap_score(question_terms, text_terms)
        metadata_score = overlap_score(question_terms, metadata_terms)
        score = chunk.similarity + (0.08 * lexical_score) + (0.04 * metadata_score)
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title=chunk.title,
                section=chunk.section,
                strategy=chunk.strategy,
                text=chunk.text,
                score=score,
                similarity=chunk.similarity,
                lexical_score=lexical_score,
                metadata_score=metadata_score,
            )
        )
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def rewrite_query(question: str) -> str:
    terms = tokenize_for_rag(question)
    expansions: list[str] = []
    aliases = {
        "mcp": ["mcp_config", "mcpServers", "/mcp"],
        "config": ["configuration", "mcp.json"],
        "конфиг": ["config", "mcp.json"],
        "индекс": ["document_index", "index_documents", "index-docs"],
        "индекса": ["document_index", "index_documents", "index-docs"],
        "документ": ["document_index", "index_documents"],
        "документов": ["document_index", "index_documents"],
        "chunk": ["chunking", "chunks"],
        "чанк": ["chunk", "chunking"],
        "чанка": ["chunk", "chunking"],
        "overlap": ["chunk_overlap", "tokens"],
        "ollama": ["nomic-embed-text", "embeddings"],
        "embedding": ["embeddings", "nomic-embed-text"],
        "embeddings": ["nomic-embed-text"],
        "scheduler": ["scheduler_storage", "scheduler.db"],
        "pipeline": ["pipeline_mcp_server", "pipeline_service"],
        "rag": ["rag_search", "rag_answer", "rag_compare"],
    }
    for term in terms:
        for value in aliases.get(term, []):
            if value.lower() not in {item.lower() for item in expansions}:
                expansions.append(value)
    if not expansions:
        return question
    return f"{question}\n\nSearch aliases: {' '.join(expansions)}"


def tokenize_for_rag(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-zА-Яа-я0-9_./~-]{3,}", text.lower()))
    return {token.strip(".,:;()[]{}\"'`") for token in tokens if token.strip(".,:;()[]{}\"'`")}


def overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


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
    baseline_matched_expected_sources = 0
    rag_term_hits = 0
    baseline_rag_term_hits = 0
    no_rag_term_hits = 0
    answers_with_sources = 0
    answers_with_quotes = 0
    aligned_answers = 0
    total_terms = 0
    for result in results:
        source_hits = result.get("source_hits")
        if isinstance(source_hits, dict):
            total_expected_sources += len(source_hits)
            matched_expected_sources += sum(1 for matched in source_hits.values() if matched)
        baseline_source_hits = result.get("baseline_source_hits")
        if isinstance(baseline_source_hits, dict):
            baseline_matched_expected_sources += sum(1 for matched in baseline_source_hits.values() if matched)
        if run_answers:
            with_rag_hits = result.get("with_rag_term_hits")
            baseline_rag_hits = result.get("baseline_rag_term_hits")
            without_rag_hits = result.get("without_rag_term_hits")
            if isinstance(with_rag_hits, dict):
                total_terms += len(with_rag_hits)
                rag_term_hits += sum(1 for matched in with_rag_hits.values() if matched)
            if isinstance(baseline_rag_hits, dict):
                baseline_rag_term_hits += sum(1 for matched in baseline_rag_hits.values() if matched)
            if isinstance(without_rag_hits, dict):
                no_rag_term_hits += sum(1 for matched in without_rag_hits.values() if matched)
            if result.get("with_rag_has_sources"):
                answers_with_sources += 1
            if result.get("with_rag_has_quotes"):
                answers_with_quotes += 1
            alignment = result.get("with_rag_answer_quote_alignment")
            if isinstance(alignment, dict) and alignment.get("aligned"):
                aligned_answers += 1

    summary: dict[str, Any] = {
        "enhanced_expected_source_matches": f"{matched_expected_sources}/{total_expected_sources}",
        "baseline_expected_source_matches": f"{baseline_matched_expected_sources}/{total_expected_sources}",
    }
    if run_answers:
        summary["enhanced_rag_expected_term_matches"] = f"{rag_term_hits}/{total_terms}"
        summary["baseline_rag_expected_term_matches"] = f"{baseline_rag_term_hits}/{total_terms}"
        summary["without_rag_expected_term_matches"] = f"{no_rag_term_hits}/{total_terms}"
        summary["answers_with_sources"] = f"{answers_with_sources}/{len(results)}"
        summary["answers_with_quotes"] = f"{answers_with_quotes}/{len(results)}"
        summary["answers_aligned_with_quotes"] = f"{aligned_answers}/{len(results)}"
    return summary
