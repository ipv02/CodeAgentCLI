from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from code_agent_cli.agent import CodeAgent
from code_agent_cli.rag_service import (
    DEFAULT_RAG_CANDIDATE_K,
    DEFAULT_RAG_MIN_SIMILARITY,
    DEFAULT_RAG_TOP_K,
    RAGError,
    RAGService,
    append_grounding_sections,
    render_quotes,
    render_rag_context,
    render_sources,
    weak_context_answer,
)


RAG_CHAT_SYSTEM_RULES = """
Ты режим контекстного чата CodeAgentCLI. Отвечай на русском.

Правила:
- используй найденный локальный контекст как главный источник фактов;
- учитывай историю диалога, working memory и Task state;
- сохраняй цель диалога, уже уточненные ограничения и термины;
- если контекст недостаточен, честно скажи "Не знаю" и попроси уточнить вопрос или переиндексировать документы;
- не добавляй факты без опоры на локальный контекст;
- не используй аббревиатуру RAG в пользовательском ответе; говори "локальный контекст", "поиск по базе" или "индекс документов";
- в конце ответа кратко отрази актуальное состояние задачи: goal, уточнения, ограничения/термины;
- источники и цитаты будут добавлены системой детерминированно, не выдумывай их.
""".strip()


@dataclass(frozen=True)
class RAGChatTurnResult:
    answer: str
    retrieval: dict[str, Any] | None
    sources: list[dict[str, Any]]
    quotes: list[dict[str, Any]]
    grounding_status: str
    best_similarity: float


@dataclass
class RAGChatService:
    agent: CodeAgent
    rag_service: RAGService
    top_k: int = DEFAULT_RAG_TOP_K
    candidate_k: int = DEFAULT_RAG_CANDIDATE_K
    min_similarity: float = DEFAULT_RAG_MIN_SIMILARITY
    response_max_tokens: int | None = None

    def send(self, user_text: str) -> RAGChatTurnResult:
        clean_text = user_text.strip()
        if not clean_text:
            raise RAGError("question не должен быть пустым.")

        try:
            retrieval = self.rag_service.search(
                clean_text,
                top_k=self.top_k,
                candidate_k=self.candidate_k,
                min_similarity=self.min_similarity,
                mode="enhanced",
            )
        except RAGError as error:
            retrieval = {
                "question": clean_text,
                "grounding_status": "weak_context",
                "best_similarity": 0.0,
                "chunks": [],
                "error": str(error),
            }
            answer = append_grounding_sections(
                f"{weak_context_answer()}\n\nПричина поиска: {error}",
                sources=[],
                quotes=[],
            )
            self.agent.save_external_turn(clean_text, answer)
            return RAGChatTurnResult(
                answer=answer,
                retrieval=retrieval,
                sources=[],
                quotes=[],
                grounding_status="weak_context",
                best_similarity=0.0,
            )
        chunks = [
            chunk
            for chunk in retrieval.get("chunks", [])
            if isinstance(chunk, dict)
        ]
        best_similarity = float(retrieval.get("best_similarity") or 0.0)
        grounding_status = (
            "grounded"
            if chunks and best_similarity >= self.min_similarity
            else "weak_context"
        )
        sources = render_sources(chunks)
        quotes = render_quotes(clean_text, chunks)

        if grounding_status == "weak_context":
            answer = append_grounding_sections(
                weak_context_answer(),
                sources=sources,
                quotes=quotes,
            )
            self.agent.save_external_turn(clean_text, answer)
            return RAGChatTurnResult(
                answer=answer,
                retrieval=retrieval,
                sources=sources,
                quotes=quotes,
                grounding_status=grounding_status,
                best_similarity=round(best_similarity, 4),
            )

        request_text = build_rag_chat_request(
            user_text=clean_text,
            agent=self.agent,
            retrieved_chunks=chunks,
        )

        def add_grounding(answer: str) -> str:
            return append_grounding_sections(
                strip_duplicate_grounding_sections(answer),
                sources=sources,
                quotes=quotes,
            )

        answer = self.agent.send_prepared_message(
            request_text=request_text,
            user_text=clean_text,
            history_text=clean_text,
            answer_postprocessor=add_grounding,
            response_max_tokens=self.response_max_tokens,
            enforce_task_lifecycle=False,
        )
        return RAGChatTurnResult(
            answer=answer,
            retrieval=retrieval,
            sources=sources,
            quotes=quotes,
            grounding_status=grounding_status,
            best_similarity=round(best_similarity, 4),
        )


def build_rag_chat_request(
    *,
    user_text: str,
    agent: CodeAgent,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    memory = agent.memory
    task_state = memory.task_state.to_dict()
    working_memory = memory.working
    long_term_memory = memory.long_term
    return "\n\n".join(
        [
            RAG_CHAT_SYSTEM_RULES,
            "Task state:\n" + format_mapping(task_state),
            "Working memory:\n" + format_mapping(working_memory),
            "Long-term memory:\n" + format_mapping(long_term_memory),
            "Локальный контекст из базы документов:\n" + render_rag_context(retrieved_chunks),
            "Новое сообщение пользователя:\n" + user_text,
        ]
    )


def format_mapping(value: dict[str, Any]) -> str:
    if not value:
        return "- нет данных"
    lines: list[str] = []
    for key in sorted(value):
        item = str(value[key]).strip()
        if item:
            lines.append(f"- {key}: {item}")
    return "\n".join(lines) if lines else "- нет данных"


def strip_duplicate_grounding_sections(answer: str) -> str:
    markers = ("Verified Sources:", "Verified Quotes:")
    cut_at = len(answer)
    for marker in markers:
        index = answer.find(marker)
        if index >= 0:
            cut_at = min(cut_at, index)
    return answer[:cut_at].strip()


RAG_CHAT_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "new_developer_onboarding",
        "goal_terms": [
            ["запуск", "перв", "старт"],
            ["истори", "history"],
            ["индекс", "sqlite", "embedding", "ollama"],
            ["источник", "sources", "quotes", "цитат"],
        ],
        "messages": [
            "Я новый разработчик в проекте. Помоги разобраться, как пользоваться этим CLI.",
            "Моя цель — понять запуск, историю диалога и работу с локальным индексом документов.",
            "Сначала объясни, как запустить обычный чат.",
            "Теперь уточнение: мне важно, где хранится история диалога.",
            "Запомни ограничение: отвечай только по документации проекта и показывай источники.",
            "Какая локальная модель используется для embedding-поиска?",
            "Где хранится SQLite-индекс документов?",
            "Что делать, если локальный индекс пустой?",
            "Проверь, что моя цель всё ещё та же.",
            "Сделай краткий итог: что я должен знать для первого запуска и поиска по документам.",
        ],
    },
    {
        "name": "requirements_brief",
        "goal_terms": [
            ["требован", "памятк"],
            ["источник", "chunk", "цитат"],
            ["огранич", "контекст"],
        ],
        "messages": [
            "Я собираю требования к помощнику, который отвечает по локальным документам проекта.",
            "Цель диалога — понять, какие данные нужны для ответа с источниками.",
            "Уточнение: пользователь не должен видеть технические детали без необходимости.",
            "Зафиксируй термин: источник — это файл, секция и chunk_id найденного фрагмента.",
            "Какие шаги выполняются перед тем, как модель отвечает пользователю?",
            "Как система понимает, что найденный контекст слабый?",
            "Что должно быть в ответе, если подходящих фрагментов нет?",
            "Ограничение: нельзя выдумывать факты без найденного контекста.",
            "Проверь, что цель и ограничения не изменились.",
            "Собери финальную памятку требований к такому помощнику.",
        ],
    },
]


def validate_rag_chat_transcript(
    scenario: dict[str, Any],
    answers: list[str],
    turn_results: list[RAGChatTurnResult],
    final_memory: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        str(message)
        for message in scenario.get("messages", [])
    ]
    goal_term_groups = normalize_goal_term_groups(scenario.get("goal_terms", []))
    failures: list[str] = []
    if len(answers) != len(messages):
        failures.append(
            f"answers count mismatch: expected {len(messages)}, got {len(answers)}"
        )

    for index, answer in enumerate(answers, start=1):
        lowered = answer.lower()
        if "verified sources:" not in lowered:
            failures.append(f"turn {index}: missing Verified Sources")
        if "verified quotes:" not in lowered:
            failures.append(f"turn {index}: missing Verified Quotes")

    for index, result in enumerate(turn_results, start=1):
        if result.grounding_status != "grounded":
            failures.append(f"turn {index}: weak local context")
        if not result.sources:
            failures.append(f"turn {index}: empty sources")
        if not result.quotes:
            failures.append(f"turn {index}: empty quotes")
        if result.retrieval and result.retrieval.get("error"):
            failures.append(f"turn {index}: retrieval error: {result.retrieval['error']}")

    tail = "\n".join(answers[-3:]).lower()
    memory_text = str(final_memory).lower()
    goal_surface = f"{tail}\n{memory_text}"
    for group in goal_term_groups:
        if not any(term in goal_surface for term in group):
            failures.append(f"goal term lost in final turns: {' | '.join(group)}")

    return {
        "scenario": scenario.get("name", "unknown"),
        "messages": len(messages),
        "answers": len(answers),
        "grounded_turns": sum(1 for result in turn_results if result.grounding_status == "grounded"),
        "ok": not failures,
        "failures": failures,
    }


def normalize_goal_term_groups(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    groups: list[list[str]] = []
    for item in value:
        if isinstance(item, list):
            terms = [str(term).lower() for term in item if str(term).strip()]
        else:
            terms = [str(item).lower()] if str(item).strip() else []
        if terms:
            groups.append(terms)
    return groups


def run_rag_chat_production_check(
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    response_max_tokens = env_optional_int("CODE_AGENT_RAG_CHAT_CHECK_MAX_TOKENS", 700)
    with TemporaryDirectory(prefix="code-agent-rag-chat-check-") as temp_dir:
        base_path = Path(temp_dir)
        for scenario in RAG_CHAT_SCENARIOS:
            scenario_name = str(scenario["name"])
            if progress_callback is not None:
                progress_callback(f"{scenario_name}: start")
            agent = CodeAgent(
                history_file=base_path / f"{scenario_name}-history.json",
                profile_file=base_path / f"{scenario_name}-profile.md",
                invariants_file=base_path / f"{scenario_name}-invariants.md",
                max_history_messages=30,
            )
            chat = RAGChatService(
                agent=agent,
                rag_service=RAGService(),
                response_max_tokens=response_max_tokens,
            )
            answers: list[str] = []
            turn_results: list[RAGChatTurnResult] = []
            failures: list[str] = []
            for index, message in enumerate(scenario.get("messages", []), start=1):
                if progress_callback is not None:
                    progress_callback(f"{scenario_name}: turn {index} request")
                try:
                    result = chat.send(str(message))
                except Exception as error:
                    failures.append(f"turn {index}: {type(error).__name__}: {error}")
                    break
                answers.append(result.answer)
                turn_results.append(result)
                if progress_callback is not None:
                    context_status = "ok" if result.grounding_status == "grounded" else "weak"
                    progress_callback(
                        (
                            f"{scenario_name}: turn {index} "
                            f"context={context_status} sources={len(result.sources)} "
                            f"quotes={len(result.quotes)}"
                        )
                    )
                if result.retrieval and result.retrieval.get("error"):
                    break

            validation = validate_rag_chat_transcript(
                scenario,
                answers,
                turn_results,
                agent.context_report(),
            )
            if failures:
                validation["failures"] = [*failures, *validation["failures"]]
                validation["ok"] = False
            results.append(validation)
    return {
        "scenarios": len(results),
        "ok": all(result["ok"] for result in results),
        "results": results,
    }


def env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else None
