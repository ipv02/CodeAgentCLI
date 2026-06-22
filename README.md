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
/mcp help
```

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
