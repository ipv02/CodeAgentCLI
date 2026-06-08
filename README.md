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
/reset
/exit
/quit
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
export CODE_AGENT_MAX_FILE_BYTES="122880"
```

Текущие настройки:

```text
/status
```

## Разработка

```bash
python -m pip install -e .
python -m compileall code_agent_cli
```
