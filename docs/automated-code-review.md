# Автоматическое AI-ревью кода

CodeAgentCLI анализирует Pull Request через GitHub Actions. Workflow запускает
`agent --review-pr`, получает changed files и Git diff, строит отдельный RAG-
индекс документации и кода, затем публикует Markdown-комментарий в PR.

## Поток данных

```text
pull_request
→ checkout доверенного review tool из base SHA
→ отдельный checkout head как недоверенных данных с полной Git-историей
→ changed files + diff base/head
→ Ollama embeddings (`nomic-embed-text`)
→ RAG по README/docs/AGENTS.md и изменённому исходному коду
→ структурированный DeepSeek review
→ проверка JSON
→ обновление комментария Pull Request
```

Review-комментарий содержит потенциальные баги, архитектурные проблемы,
рекомендации, сведения о base/head и детерминированный список RAG-источников.

Review-индекс не сканирует весь репозиторий: в него входят README, `AGENTS.md`,
`docs/`, `project/docs/` и поддерживаемые changed files. Chunks отправляются в
современный Ollama `/api/embed` пакетами по 32, а для старых версий Ollama
сохраняется fallback на одиночный `/api/embeddings`.

## GitHub

В repository secrets должен существовать `DEEPSEEK_API_KEY`. Workflow использует
минимальные разрешения `contents: read` и `pull-requests: write` и запускается
только для non-draft PR из веток текущего репозитория. Fork и Dependabot PR
пропускаются: GitHub не передает им repository secrets, а их `GITHUB_TOKEN`
имеет read-only permissions.

Пакет `CodeAgentCLI` устанавливается из base SHA. Код из PR checkout не
исполняется: он используется только как вход для Git diff и локального индекса.

GitHub-hosted runner создаётся заново для каждого запуска. Поэтому Python 3.14,
CodeAgentCLI и Ollama подготавливаются снова, а `nomic-embed-text` загружается в
локальное хранилище нового runner. Pip cache включён, но Ollama model storage и
review-индекс пока не переносятся между workflow runs.

Комментарий помечается `<!-- code-agent-cli-ai-review -->`. При новом commit в
PR workflow находит этот marker и обновляет существующий комментарий.

## Локальная проверка

Однократная подготовка embedding-модели, если она ещё не загружена:

```bash
ollama pull nomic-embed-text
```

Обычный локальный запуск:

```bash
ollama serve
export DEEPSEEK_API_KEY="ваш_ключ"

agent --review-pr \
  --review-base main \
  --review-head HEAD \
  --review-output ai-review.md
```

Локальный review-индекс хранится отдельно в
`~/.code-agent-cli/review/document_index.db`. Папку можно изменить через
`CODE_AGENT_REVIEW_DIR`.

## Безопасность и ограничения

- Git refs проверяются до передачи в subprocess; shell interpolation не
  используется.
- Индексация tracked symlink запрещена, чтобы PR не мог прочитать файл за
  пределами checkout.
- Diff, комментарии, код и RAG Evidence считаются недоверенными данными.
- Размер diff, число файлов, индекс и Evidence ограничены.
- Неизменённые исходники не пересчитываются при каждом PR review.
- Ответ модели должен соответствовать JSON-схеме; невалидный ответ завершает
  pipeline с понятной ошибкой.
- Секрет API не выводится и не сохраняется.
- Diff, измененный код и RAG Evidence передаются DeepSeek. Pull Request не должен
  содержать секреты или данные, запрещенные к передаче внешнему провайдеру.
