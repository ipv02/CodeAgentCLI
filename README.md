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
code-agent
```

Команды внутри чата:

```text
/help
/status
/tokens
/tokens проверь этот запрос без отправки
/memory
/reset
/exit
/quit
```

История диалога сохраняется между запусками в:

```text
~/.code-agent-cli/history.json
```

При следующем запуске `code-agent` загрузит сохраненные сообщения и продолжит диалог с прошлым контекстом. Команда `/reset` очищает и текущую, и сохраненную историю.

Долговременная память профиля хранится отдельно:

```text
~/.code-agent-cli/profile.md
```

Одноразовый запрос:

```bash
code-agent "объясни, чем struct отличается от class в Swift"
```

## Работа с файлами

Весь файл:

```bash
code-agent --file Sources/App.swift "найди ошибки"
```

Диапазон строк:

```bash
code-agent --file Sources/App.swift --range 40:120 "проверь участок"
```

Большой файл без подтверждения:

```bash
code-agent --file big_file.py --force-file "проверь файл"
```

Другой лимит размера файла:

```bash
code-agent --file big_file.py --max-file-bytes 200000 "проверь файл"
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
/memory
/memory short
/memory working
/memory long
/memory set working current_task реализовать memory layers
/memory set long preferences сначала объяснять архитектуру
/memory delete working current_task
/memory clear short
/memory clear working
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
сохраняется отдельно в Markdown-файле `profile.md`. Такой профиль можно
просматривать и редактировать вручную. Если в старой истории встречается
`memory.long_term`, агент не переносит его в профиль и удаляет из JSON при
следующем сохранении истории.

По умолчанию агент сам не решает, что сохранять в `working` или `long-term`.
Пользователь явно выбирает слой командой:

```text
/memory set working current_task реализовать memory layers
/memory set working files agent.py, context.py, storage.py
/memory set long preferences сначала объяснять архитектуру, потом писать код
/memory set long project_decisions CodeAgentCLI запускается через code-agent
```

Удаление также явное:

```text
/memory delete working files
/memory delete long preferences
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

Если нужен экспериментальный автоматический memory router через LLM, его можно
включить отдельно:

```bash
export CODE_AGENT_AUTO_MEMORY="1"
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
CODE_AGENT_CONTEXT_STRATEGY=sliding code-agent
CODE_AGENT_CONTEXT_STRATEGY=memory code-agent
CODE_AGENT_CONTEXT_STRATEGY=branching code-agent
```

`/context` показывает текущую стратегию, активную ветку, memory layers и токены prompt
для текущей стратегии по сравнению со sliding window. `/branch compare A B`
показывает разницу между ветками: последний user prompt, смысл ветки, prompt
tokens и отличающиеся memory layers.

## Shortcut

Чтобы запускать `code-agent` без ручной активации `.venv`, добавьте в `~/.zshrc`:

```bash
code-agent() {
  /path/to/CodeAgentCLI/.venv/bin/code-agent "$@"
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

Если `code-agent` запускается из уже установленного пакета и не видит новые команды,
переустановите локальную версию:

```bash
python -m pip install -e .
```
