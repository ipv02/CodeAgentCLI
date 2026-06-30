from __future__ import annotations

from typing import Any


RAG_EVAL_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "Где хранится MCP config?",
        "expected": "Ответ должен упомянуть ~/.code-agent-cli/mcp.json и назначение MCP config.",
        "expected_terms": ["~/.code-agent-cli/mcp.json", "MCP config"],
        "expected_sources": ["README.md", "code_agent_cli/mcp_config.py"],
    },
    {
        "question": "Какая команда строит локальный индекс документов?",
        "expected": "Ответ должен назвать /mcp index-docs PATH и связать ее с pipeline MCP.",
        "expected_terms": ["/mcp index-docs", "pipeline"],
        "expected_sources": ["README.md", "code_agent_cli/main.py"],
    },
    {
        "question": "Какая Ollama модель используется для embeddings?",
        "expected": "Ответ должен назвать nomic-embed-text и объяснить, что это embedding-модель.",
        "expected_terms": ["nomic-embed-text", "embeddings"],
        "expected_sources": ["README.md", "code_agent_cli/document_index.py"],
    },
    {
        "question": "Какие метаданные сохраняются для каждого чанка?",
        "expected": "Ответ должен перечислить source, title, section, chunk_id и strategy.",
        "expected_terms": ["source", "title", "section", "chunk_id", "strategy"],
        "expected_sources": ["README.md", "code_agent_cli/document_index.py"],
    },
    {
        "question": "Какие две стратегии chunking реализованы?",
        "expected": "Ответ должен сравнить fixed chunking и structural chunking.",
        "expected_terms": ["fixed", "structural", "chunking"],
        "expected_sources": ["README.md", "code_agent_cli/document_index.py"],
    },
    {
        "question": "Какой размер чанка и overlap используются по умолчанию?",
        "expected": "Ответ должен упомянуть 700 токенов для чанка и 80 токенов overlap.",
        "expected_terms": ["700", "80", "tokens"],
        "expected_sources": ["README.md", "code_agent_cli/document_index.py"],
    },
    {
        "question": "Где сохраняется локальный индекс документов?",
        "expected": "Ответ должен назвать document_index.db и document_index_report.json в pipeline directory.",
        "expected_terms": ["document_index.db", "document_index_report.json"],
        "expected_sources": ["README.md", "code_agent_cli/document_index.py"],
    },
    {
        "question": "Какая команда сравнивает fixed и structural chunking?",
        "expected": "Ответ должен назвать /mcp compare-chunking.",
        "expected_terms": ["/mcp compare-chunking", "fixed", "structural"],
        "expected_sources": ["README.md", "code_agent_cli/main.py"],
    },
    {
        "question": "Где scheduler хранит свои данные?",
        "expected": "Ответ должен упомянуть ~/.code-agent-cli/scheduler.db.",
        "expected_terms": ["~/.code-agent-cli/scheduler.db", "SQLite"],
        "expected_sources": ["README.md", "code_agent_cli/scheduler_storage.py"],
    },
    {
        "question": "Какие инструменты есть у pipeline MCP для RAG и индексации?",
        "expected": "Ответ должен упомянуть index_documents, rag_search, rag_answer и rag_compare.",
        "expected_terms": ["index_documents", "rag_search", "rag_answer", "rag_compare"],
        "expected_sources": ["code_agent_cli/pipeline_mcp_server.py", "code_agent_cli/pipeline_service.py"],
    },
]
