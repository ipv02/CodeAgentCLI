# CodeAgentCLI

Python CLI-ассистент для работы с кодом в терминале.

## Требования

- Python 3.9+
- Git
- API-ключ DeepSeek

## Установка

```bash
git clone git@github.com:ipv02/CodeAgentCLI.git
cd CodeAgentCLI

python3 -m venv .venv
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
/reset
/exit
/quit
```

История диалога сохраняется между запусками в:

```text
~/.code-agent-cli/history.json
```

При следующем запуске `agent` загрузит сохраненные сообщения и продолжит диалог с прошлым контекстом. Команда `/reset` полностью очищает агента: историю, память, профиль и ветки.

Долговременная память профиля хранится отдельно:

```text
~/.code-agent-cli/profile.md
```

Одноразовый запрос:

```bash
agent "объясни, чем struct отличается от class в Swift"
```

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
long-term memory
working memory
последние N сообщений как есть
текущий запрос
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

Task state сохраняется в `history.json` вместе с рабочей памятью и автоматически
подключается к каждому запросу в стратегии `memory`.

По умолчанию агент сам обновляет `task_state` внутри рабочего цикла.
Команды `/task ...` остаются как просмотр и ручной override.

Команды:

```text
/task
/task set stage planning
/task set stage execution
/task set step реализовать state machine
/task set expected проверить compileall и smoke сценарий
/task set summary задача про формализованное состояние агента
/task pause
/task resume
/task clear
```

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
Там же отображается путь к файлу истории, была ли история загружена при старте,
текущая стратегия контекста, активная ветка, memory layers и остаток контекста модели.

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
