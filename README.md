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
/reset
/exit
/quit
```

История диалога сохраняется между запусками в:

```text
~/.code-agent-cli/history.json
```

При следующем запуске `code-agent` загрузит сохраненные сообщения и продолжит диалог с прошлым контекстом. Команда `/reset` очищает и текущую, и сохраненную историю.

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
```

/tokens показывает текущую историю и последний запрос. `/tokens текст запроса`
считает токены для нового запроса без отправки в модель.

Если прогноз превышает лимит контекста, CLI не отправляет запрос в модель и
показывает, сколько токенов нужно убрать. История при этом не изменяется:
неудачный user prompt не сохраняется как часть диалога.

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
export CODE_AGENT_CONTEXT_LIMIT="64000"
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
текущий размер истории в токенах и остаток контекста модели.

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
