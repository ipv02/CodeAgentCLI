# CodeAgentCLI

Python CLI-ассистент для работы с кодом в терминале.

## Требования

- Python 3.14+
- Git
- API-ключ DeepSeek

## Установка

```bash
git clone git@github.com:ipv02/CodeAgentCLI.git
cd CodeAgentCLI

python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

HTTPS-вариант:

```bash
git clone https://github.com/ipv02/CodeAgentCLI.git
```

## API-ключ

На текущую сессию терминала:

```bash
export DEEPSEEK_API_KEY="ваш_ключ"
```

Постоянно для `zsh`:

```bash
echo 'export DEEPSEEK_API_KEY="ваш_ключ"' >> ~/.zshrc
source ~/.zshrc
```

Не храните реальный ключ в файлах проекта.

## Запуск

```bash
source .venv/bin/activate
agent
```

Команды внутри чата:

```text
/help
/status
/tokens
/tokens проверь этот запрос без отправки
/task
/memory
/profile
/invariants
/reset
/exit
/quit
```

История диалога сохраняется между запусками в:

```text
~/.code-agent-cli/history.json
```

При следующем запуске `agent` загрузит сохраненные сообщения и продолжит диалог с прошлым контекстом. Команда `/reset` очищает историю, память, профиль и ветки. Инварианты сохраняются отдельно и очищаются только командой `/invariants clear`.

Долговременная память профиля хранится отдельно:

```text
~/.code-agent-cli/profile.md
```

Обязательные инварианты ассистента хранятся отдельно от диалога и профиля:

```text
~/.code-agent-cli/invariants.md
```

Одноразовый запрос:

```bash
agent "объясни, чем struct отличается от class в Swift"
```

## MCP

CLI подключается к реальным MCP stdio-серверам из постоянного config и выводит
список инструментов, которые эти серверы объявляют.

Постоянный MCP config хранится в:

```text
~/.code-agent-cli/mcp.json
```

При старте `agent` показывает, настроен ли MCP:

```text
Модель: deepseek-v4-flash · ... · MCP: 2 configured
```

Добавить свой MCP-сервер один раз:

```text
agent
/mcp add NAME -- COMMAND ARG1 ARG2
```

Примеры:

```text
/mcp add apple-mcp -- bunx --no-cache apple-mcp@latest
/mcp add cupertino -- cupertino serve --no-reap
```

После добавления сервер остается в config и подхватывается при следующих запусках
`agent`.

Удалить один MCP-сервер:

```text
/mcp remove NAME
```

Полностью отключить MCP:

```text
/mcp clear
```

Команды `/mcp add`, `/mcp remove` и `/mcp clear` сами обновляют JSON config.
Пользователю не нужно вручную редактировать скобки, запятые и массивы.

Проверить подключение:

```text
/mcp
```

Ожидаемый статус:

```text
MCP:
Конфиг: /Users/ipv/.code-agent-cli/mcp.json
Серверов: 2

  apple-mcp  Connected  7 инструментов
  cupertino  Connected  15 инструментов

Connected servers: 2 / 2
Инструментов: 22
```

Показать полный список инструментов:

```text
/mcp tools
```

Для одноразовой проверки из shell без входа в интерактивный режим:

```bash
agent --mcp-config-tools
```

Дополнительные команды:

```text
/mcp show
/mcp remove NAME
/mcp clear
/mcp path
/mcp test
/mcp init-scheduler
/mcp help
```

### Локальная индексация документов через Ollama

Pipeline MCP умеет строить локальный индекс документов для RAG-сценариев:
README, Markdown/TXT/RST, Python-код, JSON/YAML/TOML и PDF при установленном
`pypdf`.

Перед индексацией запустите Ollama и скачайте embedding-модель:

```bash
ollama serve
ollama pull nomic-embed-text
```

### Локальный чат через Ollama

Для локального чата без внешнего LLM API можно использовать Ollama-модель.
По умолчанию CodeAgentCLI использует `llama3.2:3b`:

```bash
ollama serve
ollama pull llama3.2:3b
agent --local-chat
```

Быстрая проверка перед запуском чата:

```bash
ollama list
ollama run llama3.2:3b "Сколько будет 17 * 23? Ответь кратко."
```

Проверка HTTP API:

```bash
curl http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Ответь одним предложением: что такое локальная LLM?"}],"stream":false}'
```

Выбрать другую модель можно флагом или переменной окружения:

```bash
agent --local-chat --local-model qwen2.5-coder:3b
CODE_AGENT_LOCAL_MODEL=qwen2.5-coder:3b agent --local-chat
```

Команды внутри режима:

```text
/model   показать локальную модель и адрес Ollama
/reset   очистить историю текущего локального чата
/pull    показать команду скачивания текущей модели
/help
/exit
```

`agent --local-chat` обращается к локальному Ollama API
`http://127.0.0.1:11434` и не требует `DEEPSEEK_API_KEY`. Это обычный чат с
локальной моделью. Для ответов с поиском по локальной базе документов и
локальной генерацией используйте `agent --local-context-chat`.

Параметры обычного локального чата передаются в Ollama явно:

```bash
CODE_AGENT_LOCAL_TEMPERATURE=0.2 \
CODE_AGENT_LOCAL_NUM_PREDICT=512 \
CODE_AGENT_LOCAL_NUM_CTX=4096 \
agent --local-chat
```

`num_predict` ограничивает число токенов ответа, а `num_ctx` задает контекстное
окно запроса.

Пример ручной проверки внутри `agent --local-chat`:

```text
Ответь одним предложением: что такое локальная LLM?
Объясни в 3 пунктах, зачем нужен health check для CLI-приложения.
Напиши короткую Python-функцию is_even(n: int) -> bool и один пример использования.
/model
/reset
/exit
```

### Приватный LLM-сервис по HTTP

`agent --llm-service` поднимает приватный HTTP gateway к локальной Ollama-модели.
Ollama остается backend-сервисом, а наружу публикуется API CodeAgentCLI с auth,
rate limit и ограничениями контекста.

Локальный запуск только на этой машине:

```bash
agent --llm-service
```

Запуск на VPS или домашнем сервере для доступа по сети:

```bash
CODE_AGENT_LLM_SERVICE_API_KEY='replace-with-private-token' \
agent --llm-service --llm-service-host 0.0.0.0 --llm-service-port 8080
```

Если `--llm-service-host` не loopback-адрес, API key обязателен. Это защищает от
случайной публикации локальной модели без авторизации.

Основные endpoints:

```text
GET  /chat
GET  /health
GET  /v1/models
POST /v1/chat
POST /v1/chat/completions
```

Браузерный чат:

```text
http://SERVER_IP:8080/chat
```

Если сервис запущен с `CODE_AGENT_LLM_SERVICE_API_KEY`, введите этот token в
поле `Bearer token`. История чата хранится в текущей вкладке браузера и
отправляется в `/v1/chat` вместе с новым сообщением.

Проверка с другой машины:

```bash
curl http://SERVER_IP:8080/health

curl http://SERVER_IP:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-private-token' \
  -d '{"messages":[{"role":"user","content":"Ответь одним предложением: что такое приватная LLM?"}]}'
```

OpenAI-compatible форма:

```bash
curl http://SERVER_IP:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-private-token' \
  -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Привет"}],"temperature":0,"max_tokens":128}'
```

Лимиты сервиса:

```bash
CODE_AGENT_LLM_SERVICE_RATE_LIMIT=30 \
CODE_AGENT_LLM_SERVICE_MAX_BODY_BYTES=131072 \
CODE_AGENT_LLM_SERVICE_MAX_MESSAGES=32 \
CODE_AGENT_LLM_SERVICE_MAX_MESSAGE_CHARS=16000 \
CODE_AGENT_LOCAL_NUM_CTX=4096 \
CODE_AGENT_LOCAL_NUM_PREDICT=512 \
agent --llm-service
```

`CODE_AGENT_LOCAL_NUM_CTX` задает максимальное контекстное окно, а
`CODE_AGENT_LOCAL_NUM_PREDICT` ограничивает длину ответа. Запросы с `num_ctx` или
`max_tokens` выше лимита сервиса отклоняются.

Для стабильного production-запуска обычно держат Ollama на `127.0.0.1:11434`,
а `agent --llm-service` публикуют через firewall или reverse proxy с HTTPS.

Подключить pipeline MCP и построить индекс проекта:

```text
agent
/mcp init-pipeline
/mcp index-docs .
/mcp index-status
/mcp compare-chunking
```

Индекс сохраняется локально:

```text
~/.code-agent-cli/pipeline/document_index.db
~/.code-agent-cli/pipeline/document_index_report.json
```

Для каждого чанка сохраняются embedding-вектор Ollama и метаданные:
`source`, `title`, `section`, `chunk_id`, `strategy`, `start_char`, `end_char`.
Отчет сравнивает две стратегии chunking: фиксированный размер и структурное
разбиение по заголовкам, файлам, классам и функциям. Размер чанка задается
в токенах: по умолчанию `700` токенов на чанк и `80` токенов overlap.
Допустимые значения: `chunk_size` от `500` до `1000`, `overlap` от `50` до
`100`.

### Контекстный чат по локальной базе документов

После построения индекса можно запустить отдельный CLI-режим, который перед
каждым ответом ищет релевантные фрагменты в локальной базе документов:

```bash
agent --context-chat
```

На старте режим показывает:

```text
Code Agent CLI · Контекстный чат
Каждый вопрос ищет фрагменты в локальной базе документов.
Ollama строит embedding вопроса, затем ответ собирается с источниками.
```

Поток одного сообщения:

```text
вопрос пользователя
  -> Ollama embedding
  -> поиск похожих chunks в SQLite index
  -> prompt: история диалога + task state + найденный контекст + вопрос
  -> DeepSeek answer
  -> компактный ответ + источники + цитаты
```

Контекстный чат хранит историю диалога через обычный `HistoryStorage`, а
текущую цель, уточнения, ограничения и термины — через существующие
`TaskState` и working memory. В prompt модели попадает только подготовленный
контекст; найденные chunks не загрязняют память задачи.

Команды внутри режима:

```text
/state          показать цель, текущий шаг, ограничения/термины и последние сообщения
/sources        показать источники и цитаты последнего ответа
/reset-context  очистить историю текущего диалога и task memory
/tokens         показать токены и стоимость последнего запроса
/status         показать статус агента
/exit
```

Обычный ответ разделен на читаемые блоки:

```text
== Контекст найден ==
similarity: 0.7165
sources: 5
quotes: 5

Ответ:
  ...

Память задачи:
stage: planning
goal: ...
step: ...
next: ...

Источники:
  1. README.md
    section
    chunk_id · similarity=...

Цитаты:
  1. README.md
    короткий фрагмент из найденного chunk
```

Если локальный контекст слабый, режим отвечает `Не знаю`, просит уточнить
вопрос или переиндексировать документы и все равно показывает блоки источников
и цитат.

Проверить режим на двух длинных production-like сценариях:

```bash
agent --context-chat-check
```

Проверка запускает реальные ответы через локальный SQLite index, Ollama
embedding-модель и DeepSeek. Сценарии:

- `new_developer_onboarding`: новый разработчик уточняет запуск CLI, историю
  диалога, embedding-модель и SQLite index;
- `requirements_brief`: аналитик собирает требования к помощнику по локальным
  документам, фиксирует термин `источник`, ограничение на выдумывание фактов и
  проверяет удержание цели.

Ожидаемый итог:

```text
Проверка контекстного чата:
Scenarios: 2
Status: OK

  new_developer_onboarding
  messages: 10
  answers: 10
  turns with context: 10
  ok: True

  requirements_brief
  messages: 10
  answers: 10
  turns with context: 10
  ok: True
```

### Первый RAG-запрос

После построения индекса pipeline MCP умеет отвечать на вопросы в двух режимах:
без RAG и с RAG-контекстом из локального SQLite index.

```text
agent
/mcp init-pipeline
/mcp index-docs /Users/ipv/Desktop/Develop/CodeAgentCLI
/mcp rag-search "Где хранится MCP config?"
/mcp rag-answer "Где хранится MCP config?"
/mcp rag-compare "Где хранится MCP config?"
/mcp rag-eval
```

RAG flow:

```text
question -> query rewrite -> Ollama embedding -> candidate chunks -> similarity filter + heuristic rerank -> LLM answer with verified sources and quotes
```

По умолчанию RAG использует enhanced retrieval без отдельной reranker-модели:
сначала берет `candidate_k` chunks, затем отсекает результаты ниже
`min_similarity` и переупорядочивает оставшиеся chunks простыми эвристиками
по совпадению терминов в тексте, title, section и source.

RAG-ответы дополнительно возвращают проверяемое grounding-представление:

```text
Answer: ответ на вопрос
Verified Sources: source / section / chunk_id
Verified Quotes: короткие фрагменты из найденных chunks
```

Цитаты строятся детерминированно из найденных chunks, а не придумываются
моделью. Если после фильтрации нет достаточно релевантного контекста или лучший
`similarity` ниже `min_similarity`, RAG отвечает `Не знаю` и просит уточнить
вопрос.

Полностью локальный RAG-флоу:

```text
question -> query rewrite -> Ollama embedding -> local SQLite retrieval -> compact Evidence -> Ollama answer -> verified sources/quotes
```

Обычные команды `/mcp rag-search`, `/mcp rag-answer`, `agent --context-chat` и
`agent --context-chat-check` сохраняют стандартный enhanced retrieval для
облачной генерации. Локальная генерация использует отдельный local-only путь:
он добавляет hybrid retrieval поверх SQLite index и передает в Ollama компактный
блок `Evidence`, чтобы локальная модель видела не большой шумный chunk, а
короткие релевантные фрагменты с `source`, `section`, `chunk_id`, цитатами и
точными строками вроде путей, команд и имен моделей.

Запуск интерактивного режима:

```bash
agent --local-context-chat
agent --local-context-chat --local-model qwen2.5-coder:3b
```

MCP shortcut для одного локального ответа:

```text
/mcp rag-answer-local "Где хранится SQLite-индекс документов?"
```

Команды внутри локального контекстного чата:

```text
/state          показать состояние задачи и память текущего диалога
/sources        показать источники и цитаты последнего ответа
/reset-context  очистить историю текущего диалога и task memory
/help
/exit
```

Минимальная проверка полностью локального RAG:

```text
agent --local-context-chat
Где хранится SQLite-индекс документов?
Какая модель используется для embeddings?
Что делает enhanced retrieval в локальном поиске?
/sources
/exit
```

Ожидаемо: режим показывает `generation: local`, модель `llama3.2:3b` или
выбранную через `--local-model`, источники и цитаты. Для этих вопросов ответ
должен извлекаться из локального индекса, включая `document_index.db`,
`nomic-embed-text` и шаги `query rewrite`, `similarity filter`,
`heuristic rerank`.

### Оптимизация локальной модели для ответов по документам

`agent --local-rag-optimize` сравнивает два профиля генерации на контрольных
вопросах из проекта:

- `baseline`: прежний prompt, `temperature=0.2`, стандартные Ollama
  `num_predict` и `num_ctx`;
- `optimized`: строгий Evidence prompt, `temperature=0.0`,
  `num_predict=500`, `num_ctx=4096`.

Для каждого вопроса retrieval выполняется один раз. Оба профиля получают
одинаковые chunks, поэтому отчет сравнивает именно генерацию, а не разные
результаты поиска.

Быстрый наглядный прогон:

```bash
agent --local-rag-optimize --optimization-questions 1 --optimization-repeats 2
```

Полный прогон:

```bash
agent --local-rag-optimize --optimization-questions 10 --optimization-repeats 3
```

Отчет показывает:

- ответы до и после оптимизации;
- совпадения ожидаемых фактов и итоговую оценку качества;
- источники из локального SQLite-индекса;
- время retrieval и генерации;
- токены ответа и токены в секунду из Ollama API;
- стабильность ответа и качества на повторных запусках;
- размер, фактическое квантование и память загруженной модели, если Ollama
  возвращает эти данные через `/api/show` и `/api/ps`.

Настройки optimized-профиля:

```bash
CODE_AGENT_LOCAL_RAG_TEMPERATURE=0.0 \
CODE_AGENT_LOCAL_RAG_NUM_PREDICT=500 \
CODE_AGENT_LOCAL_RAG_NUM_CTX=4096 \
agent --local-rag-optimize
```

Эти же настройки использует `agent --local-context-chat`. Стандартный
retrieval, SQLite-индекс, verified sources и verified quotes не меняются.
Проверить другую уже установленную модель можно так:

```bash
agent --local-rag-optimize --local-model qwen2.5-coder:3b
```

Команда `/mcp rag-compare` показывает локальную генерацию и, если задан
`DEEPSEEK_API_KEY`, облачную генерацию на найденном контексте:

```text
Local model: ответ локальной модели без локального контекста
Local baseline RAG: локальная модель + обычный vector search без rewrite/filter/rerank
Local enhanced RAG: локальная модель + query rewrite, similarity filter и heuristic rerank
Cloud model / Cloud baseline RAG / Cloud enhanced RAG: облачное сравнение, если доступно
```

Команда `/mcp rag-eval` использует 10 контрольных вопросов по базе проекта.
Для каждого вопроса зафиксированы ожидание, ключевые термины и ожидаемые
источники. Итог показывает совпадения по источникам и ключевым терминам для
baseline RAG, enhanced RAG и non-RAG ответов, а также проверяет наличие
источников, цитат и примерное совпадение смысла ответа с цитатами. Полный
прогон делает LLM-запросы для каждого режима ответа и может занять время.

## Встроенный mock HTTP API MCP

Для первого собственного MCP-инструмента в проекте есть встроенный stdio-сервер
вокруг публичного mock API `http://jsonplaceholder.typicode.com`.

Подключить сервер:

```text
agent
/mcp init-mock
```

Проверить регистрацию инструмента и описание входных параметров:

```text
/mcp tools
```

Ожидаемый инструмент:

```text
get_mock_user
Вход: user_id: integer, required
```

Вызвать инструмент напрямую из приложения:

```text
/mcp call mock-api get_mock_user {"user_id": 1}
```

Инструмент делает HTTP-запрос:

```text
GET http://jsonplaceholder.typicode.com/users/1
```

И возвращает нормализованный результат с `id`, `name`, `email`, `company`,
`city` и ссылкой на источник.

Обычный запрос к агенту тоже может использовать MCP-результат:

```text
расскажи про mock user 1
```

В этом сценарии `agent` вызывает MCP-инструмент `mock-api/get_mock_user`,
получает результат и передает его модели как контекст для ответа.

## MCP-планировщик и фоновые задачи

В проекте есть встроенный stdio MCP-сервер `scheduler` для отложенных и
периодических agent-задач. Он хранит задачи и результаты запусков в SQLite:

```text
~/.code-agent-cli/scheduler.db
```

Подключить сервер:

```text
agent
/mcp init-scheduler
```

Проверить инструменты:

```text
/mcp tools
```

Основные tools:

```text
health
remind
every
jobs
delete
run_due
summary
```

Создать отложенное напоминание:

```text
/mcp remind "Проверить статус проекта" 2026-06-24T12:30:00Z
```

Создать периодическую сводку:

```text
/mcp every "Daily summary" 1440 "Собрать краткую сводку по проекту"
```

`every` создает периодическую LLM-задачу: при выполнении `scheduler-runner`
отправляет prompt в DeepSeek-compatible chat completions API, сохраняет ответ
модели в SQLite и затем показывает его через `summary`.

Выполнить задачи, срок которых наступил:

```text
/mcp run_due
```

Получить агрегированный результат:

```text
/mcp summary
```

Низкоуровневый JSON-вызов тоже доступен:

```text
/mcp call scheduler remind {"text":"Проверить статус проекта","run_at":"2026-06-24T12:30:00Z"}
```

Проверочный сценарий для задания:

```text
/mcp init-scheduler
/mcp remind "Проверить фоновую задачу" 2020-01-01T00:00:00Z
/mcp run_due
/mcp summary
```

Очистить все задачи и историю запусков scheduler:

```text
/mcp clear-scheduler
```

В выводе нужно проверить ключевые строки:

```text
Reminder создан
Сохранено: SQLite jobs
Due jobs: 1
Успешно: 1
Сохранено: SQLite job_runs
Сводка планировщика
Последних запусков: 1
Ошибок: 0
Результат: Напоминание: Проверить фоновую задачу
```

Для проверки LLM-сводки создайте periodic job:

```text
/mcp every "LLM summary check" 1 "Сгенерируй короткую сводку: планировщик работает 24/7 и сохраняет результаты."
```

После наступления времени запуска `scheduler-runner` выполнит задачу. В
`/mcp summary` для такого запуска должны быть строки:

```text
Источник: LLM-generated summary
Модель: deepseek-v4-flash
Результат: ...
```

В интерактивном терминале успешные статусы, ошибки, даты следующего запуска и
ключевые значения подсвечиваются цветом.

Для фонового режима установочный пакет добавляет команду `scheduler-runner`.
Одноразовый запуск подходит для cron или systemd timer:

```bash
scheduler-runner
```

Постоянный процесс 24/7:

```bash
scheduler-runner --watch --interval 60
```

В таком режиме MCP tools создают и читают задачи, SQLite хранит состояние, а
`scheduler-runner` периодически выполняет due jobs и сохраняет результаты,
которые затем возвращает `summary`.

## MCP pipeline: search -> summarize -> save

В проекте есть встроенный MCP-сервер `pipeline`, который демонстрирует
композицию нескольких MCP-инструментов:

```text
search
summarize
save
run
```

Подключить сервер:

```text
agent
/mcp init-pipeline
```

Проверить инструменты:

```text
/mcp tools
```

Запустить автоматическую цепочку:

```text
/mcp pipeline "latest Model Context Protocol news" mcp-summary.md
```

Агент также умеет сам распознавать обычную пользовательскую фразу и запускать
pipeline без ручного `/mcp call`:

```text
найди мне Model Context Protocol MCP и сохрани в заметки
```

В этом сценарии `agent` автоматически вызывает MCP tool `pipeline/run`,
который выполняет цепочку `search -> summarize -> save`, отвечает сводкой и
сохраняет результат в `notes.md`.

Что происходит внутри:

```text
search     -> ищет данные в интернете
summarize  -> передает результаты поиска в LLM и делает сводку
save       -> сохраняет результат в ~/.code-agent-cli/pipeline/
run        -> автоматически выполняет search -> summarize -> save
```

Низкоуровневый вызов:

```text
/mcp call pipeline run {"query":"latest Model Context Protocol news","filename":"mcp-summary.md","limit":5}
```

В выводе нужно проверить ключевые строки:

```text
Автоматический MCP pipeline
Цепочка: search -> summarize -> save
Results: ...
Items used: ...
Model: deepseek-v4-flash
Saved: yes
Path: ~/.code-agent-cli/pipeline/mcp-summary.md
```

## MCP orchestration: multi-server flow

Для длинных сценариев агент использует внутренний `MCPOrchestrationAgent`.
Он получает список реально зарегистрированных MCP tools, строит JSON-план,
валидирует server/tool names и выполняет шаги по порядку.

Основные серверы для orchestration:

```text
apple-mcp
cupertino
pipeline
scheduler
```

Подключить недостающие серверы:

```text
agent
/mcp init-orchestration
```

Проверить, какие tools реально доступны:

```text
/mcp tools
```

Запустить длинный flow:

```text
/mcp orchestrate "найди лучшие практики навигации SwiftUI в iOS через Cupertino MCP, сделай сводку, сохрани в заметки и поставь напоминание проверить завтра"
```

Такой же сценарий можно запустить обычной фразой без `/mcp`:

```text
найди лучшие практики навигации SwiftUI в iOS через Cupertino MCP, сделай сводку, сохрани в заметки и поставь напоминание проверить завтра
```

Ожидаемый порядок вызовов:

```text
cupertino/search
pipeline/summarize_text
pipeline/save
scheduler/remind
scheduler/summary
```

В выводе нужно проверить:

```text
MCP Orchestration
Flow: cupertino/search -> pipeline/summarize_text -> pipeline/save -> scheduler/remind -> scheduler/summary
Step 1: cupertino/search
Step 2: pipeline/summarize_text
Step 3: pipeline/save
Saved: ~/.code-agent-cli/pipeline/notes.md
Step 4: scheduler/remind
Next run: ...
Step 5: scheduler/summary
Active jobs: ...
```

Передача данных между tools выполняется через ссылки в плане:

```text
$previous_text
$steps[1]
$steps[2].summary
$tomorrow_09_utc
```

`MCPOrchestrationAgent` только планирует. Реальное выполнение делает
детерминированный runner через существующий MCP client, поэтому план не может
вызвать незарегистрированный server/tool.

Создать готовый config для Apple MCP и Cupertino:

```bash
agent --mcp-init-apple
```

Реализация использует официальный Python MCP SDK: `ClientSession`,
`StdioServerParameters`, `stdio_client`, затем `initialize()` и `list_tools()`.

Config создается в формате:

```json
{
  "mcpServers": {
    "apple-mcp": {
      "command": "bunx",
      "args": ["--no-cache", "apple-mcp@latest"]
    },
    "cupertino": {
      "command": "cupertino",
      "args": ["serve", "--no-reap"]
    }
  }
}
```

`apple-mcp` требует установленный Bun (`bunx`), а `cupertino` требует установленный
Cupertino и одноразовую настройку баз документации через `cupertino setup`.

## Работа с файлами

Весь файл:

```bash
agent --file Sources/App.swift "найди ошибки"
```

Диапазон строк:

```bash
agent --file Sources/App.swift --range 40:120 "проверь участок"
```

Большой файл без подтверждения:

```bash
agent --file big_file.py --force-file "проверь файл"
```

Другой лимит размера файла:

```bash
agent --file big_file.py --max-file-bytes 200000 "проверь файл"
```

Содержимое приложенного файла не сохраняется целиком в историю диалога.

## Токены и стоимость

CLI считает токены локально перед запросом и сверяет их с `usage`, который возвращает API:

- текущий запрос;
- вся история диалога вместе с system prompt;
- prompt tokens;
- answer tokens;
- примерная стоимость input/output.

После ответа показывается отчет по токенам: оценка текущего запроса и истории,
а также фактический `usage` от модели: prompt, answer и total.

В интерактивном режиме доступны команды:

```text
/tokens
/tokens текст запроса
/context
/context strategy sliding
/context strategy memory
/context strategy branching
/task
/task set stage execution
/task set step добавить state machine
/task set expected проверить compileall
/task pause
/task resume
/memory
/memory short
/memory working
/memory long
/memory clear short
/memory clear working
/memory clear all
/profile
/profile path
/invariants
/invariants add не менять выбранную архитектуру без явного решения пользователя
/invariants delete 1
/invariants clear
/branch list
/branch compare variant-a variant-b
/branch checkpoint base
/branch create variant-a base
/branch switch variant-a
```

/tokens показывает текущую историю и последний запрос. `/tokens текст запроса`
считает токены для нового запроса без отправки в модель.

Если прогноз превышает лимит контекста, CLI не отправляет запрос в модель и
показывает, сколько токенов нужно убрать. История при этом не изменяется:
неудачный user prompt не сохраняется как часть диалога.

## Стратегии контекста

Агент поддерживает три стратегии управления контекстом:

### Sliding Window

Хранит только последние `CODE_AGENT_MAX_HISTORY` сообщений. Всё более старое
отбрасывается.

```bash
export CODE_AGENT_CONTEXT_STRATEGY="sliding"
```

Плюсы: дешево и предсказуемо. Минус: старые детали забываются.

### Memory Layers

Агент использует явную модель памяти. Память разделена на слои:

- `short-term` — последние сообщения текущего диалога;
- `working` — данные текущей задачи: цель, план, файлы, временные ограничения и риски;
- `long-term` — профиль, предпочтения, устойчивые решения проекта и знания.

`short-term` и `working` сохраняются в `history.json`, а `long-term`
сохраняется отдельно в Markdown-файле `profile.md`. Если в старой истории встречается
`memory.long_term`, агент не переносит его в профиль и удаляет из JSON при
следующем сохранении истории.

Инварианты не являются памятью диалога. Это обязательные ограничения: выбранная
архитектура, принятые технические решения, ограничения стека или бизнес-правила.
Они подключаются к каждому запросу отдельным system-блоком и проверяются
отдельным внутренним агентом до запуска основного ответа. Если запрос конфликтует
с инвариантом, ассистент отказывается от конфликтующей части, называет нарушенное
правило и предлагает совместимую альтернативу.

По умолчанию `auto memory` включен. На каждом сообщении пользователя агент
запускает внутренний memory router и сам решает, что сохранить:

```text
- в working: текущую задачу, файлы, риски, временные ограничения
- в long-term: профиль пользователя, предпочтения, устойчивые решения и знания
- в discard: шум, приветствия и одноразовые фразы
```

Память теперь не редактируется вручную из CLI. Доступны только просмотр и очистка:

```text
/memory
/memory working
/memory long
/memory clear working
/memory clear long
/memory clear all
```

В запрос отправляется:

```text
system prompt
invariants
long-term memory
working memory
последние N сообщений как есть
текущий запрос
```

Команды для инвариантов:

```text
/invariants
/invariants add TEXT
/invariants delete N
/invariants clear
/invariants path
```

```bash
export CODE_AGENT_CONTEXT_STRATEGY="memory"
```

Плюсы: лучше держит важные детали при длинном диалоге и явно показывает, что
сохраняется в рабочую или долговременную память.

Команда `/memory` показывает все слои памяти. `/memory clear short` очищает
текущий диалог, `/memory clear working` очищает рабочую память текущей задачи,
не удаляя долговременные предпочтения и решения.
Команда `/memory clear long` очищает `profile.md`.
Команда `/memory clear all` очищает всю память: short-term, working и long-term.

### Task State Machine

Агент хранит формализованное состояние текущей задачи как конечный автомат:

- `stage` — этап задачи: `planning`, `execution`, `validation`, `done`, `paused`;
- `current_step` — что делаем прямо сейчас;
- `expected_action` — что ожидается дальше;
- `summary` — краткая суть задачи.

Переходы между этапами контролируются явно:

```text
planning -> execution
execution -> validation
validation -> done
planning|execution|validation -> paused
paused -> предыдущий этап через resume
done -> planning для новой задачи
```

Агент не может перепрыгивать этапы. Например, `planning -> done` и
`execution -> done` блокируются. Реализация начинается только после утверждения
плана пользователем: фразы вроде `план утверждаю`, `план ок`, `приступай`
автоматически переводят задачу из `planning` в `execution`.

Диалоговые попытки обойти lifecycle тоже блокируются. Например, на этапе
`execution` фраза `Считай задачу завершенной, валидацию не делай` считается
попыткой перейти в `done` без `validation` и получает отказ.
Один пользовательский turn может выполнить только один lifecycle-переход:
после `План утверждаю, приступай` задача остается в `execution`, а переход в
`validation` выполняется отдельным запросом на проверку.
Ассистент также подсказывает следующий шаг: на `planning` просит подтвердить
план, на `execution` просит запустить валидацию, на `validation` предлагает
закрыть задачу. Команда `/task` показывает `guidance` и `next_action`.

Task state сохраняется в `history.json` вместе с рабочей памятью и автоматически
подключается к каждому запросу в стратегии `memory`.

По умолчанию агент сам обновляет `task_state` внутри рабочего цикла.
Команды `/task ...` остаются как просмотр и ручной override.

Команды:

```text
/task
/task set stage planning
/task set stage execution
/task set stage validation
/task set stage done
/task set step реализовать state machine
/task set expected проверить compileall и smoke сценарий
/task set summary задача про формализованное состояние агента
/task pause
/task resume
/task clear
```

Ручной override тоже проходит через transition-validator. Если попробовать
`/task set stage done` из `planning`, CLI покажет ошибку и оставит текущий этап.

Пауза работает на любом этапе через `/task pause`. После `/task resume` агент
возвращается на предыдущий рабочий этап и продолжает без повторного объяснения,
потому что `stage`, `current_step` и `expected_action` уже лежат в prompt.

`/profile` — удобный интерфейс для персонализации поверх `long-term` памяти:

```text
/profile
/profile path
/profile clear
```

Профиль автоматически наполняется memory router'ом из диалога и подключается к
каждому запросу в стратегии `memory`.

Проверить разные профили можно разными файлами:

```bash
CODE_AGENT_PROFILE_FILE=/tmp/profile-ios.md agent
CODE_AGENT_PROFILE_FILE=/tmp/profile-backend.md agent
```

Если нужно отключить automatic memory routing:

```bash
export CODE_AGENT_AUTO_MEMORY="0"
```

### Branching

Позволяет сохранять checkpoint, создавать ветки от него и продолжать диалог в
каждой ветке независимо. Ветка хранит собственные сообщения и memory layers.

```bash
export CODE_AGENT_CONTEXT_STRATEGY="branching"
```

Команды:

```text
/branch list
/branch compare variant-a variant-b
/branch checkpoint base
/branch create variant-a base
/branch create variant-b base
/branch switch variant-a
```

Сравнить режимы можно так:

```bash
CODE_AGENT_CONTEXT_STRATEGY=sliding agent
CODE_AGENT_CONTEXT_STRATEGY=memory agent
CODE_AGENT_CONTEXT_STRATEGY=branching agent
```

`/context` показывает текущую стратегию, активную ветку, memory layers и токены prompt
для текущей стратегии по сравнению со sliding window. `/branch compare A B`
показывает разницу между ветками: последний user prompt, смысл ветки, prompt
tokens и отличающиеся memory layers.

## Shortcut

Чтобы запускать `agent` без ручной активации `.venv`, добавьте в `~/.zshrc`:

```bash
agent() {
  /path/to/CodeAgentCLI/.venv/bin/agent "$@"
}
```

Затем:

```bash
source ~/.zshrc
```

## Настройки

```bash
export CODE_AGENT_MODEL="deepseek-v4-flash"
export CODE_AGENT_TEMPERATURE="0.2"
export CODE_AGENT_MAX_HISTORY="20"
export CODE_AGENT_CONTEXT_STRATEGY="memory"
export CODE_AGENT_MEMORY_MAX_TOKENS="1200"
export CODE_AGENT_AUTO_MEMORY="0"
export CODE_AGENT_CONTEXT_LIMIT="64000"
export CODE_AGENT_PROFILE_FILE="~/.code-agent-cli/profile.md"
export CODE_AGENT_INVARIANTS_FILE="~/.code-agent-cli/invariants.md"
export CODE_AGENT_INPUT_PRICE_PER_1M="0.28"
export CODE_AGENT_OUTPUT_PRICE_PER_1M="0.42"
export CODE_AGENT_MAX_FILE_BYTES="122880"
export CODE_AGENT_HISTORY_FILE="$HOME/.code-agent-cli/history.json"
```

Текущие настройки:

```text
/status
```

`/status` также показывает токены текущей сессии: total, prompt, answer.
Там же отображается путь к файлам истории, профиля и инвариантов, была ли история
загружена при старте, текущая стратегия контекста, активная ветка, memory layers,
invariants и остаток контекста модели.

## Разработка

```bash
python -m pip install -e .
python -m compileall code_agent_cli
```

Если `agent` запускается из уже установленного пакета и не видит новые команды,
переустановите локальную версию:

```bash
python -m pip install -e .
```
