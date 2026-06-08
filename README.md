# CodeAgentCLI

Лёгкий Python CLI-ассистент для кодинга в терминале.

Умеет работать в интерактивном чате, отвечать на одноразовые запросы, читать
файлы с кодом, показывать лоадер `Думаю...`, подсвечивать ответы и оформлять
блоки кода.

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

Если SSH для GitHub не настроен:

```bash
git clone https://github.com/ipv02/CodeAgentCLI.git
```

## API-ключ

Временный вариант для текущего терминала:

```bash
export DEEPSEEK_API_KEY="ваш_реальный_deepseek_ключ"
```

Постоянный вариант для `zsh`:

```bash
echo 'export DEEPSEEK_API_KEY="ваш_реальный_deepseek_ключ"' >> ~/.zshrc
source ~/.zshrc
```

Для `bash` используйте `~/.bashrc` или `~/.bash_profile`.

Не добавляйте реальный ключ в файлы проекта и не коммитьте его в Git.

## Запуск

Интерактивный режим:

```bash
source .venv/bin/activate
code-agent
```

На старте CLI показывает модель и размер истории:

```text
Code Agent CLI
Команды: /help, /status, /reset, /exit
Модель: deepseek-v4-flash · История: 0/20
>
```

В новом терминале достаточно снова активировать `.venv`:

```bash
cd CodeAgentCLI
source .venv/bin/activate
code-agent
```

Создавать `.venv` и выполнять `pip install .` каждый раз не нужно.

## Shortcut

Чтобы запускать `code-agent` из любой папки без ручной активации `.venv`,
добавьте в `~/.zshrc`:

```bash
code-agent() {
  /path/to/CodeAgentCLI/.venv/bin/code-agent "$@"
}
```

Замените `/path/to/CodeAgentCLI` на путь к папке проекта, затем выполните:

```bash
source ~/.zshrc
```

## Команды чата

```text
/help   помощь
/status текущие настройки
/reset  очистить историю сессии
/exit   выйти
/quit   выйти
```

История хранится только в памяти текущего запуска CLI.

## Примеры

Одноразовый запрос:

```bash
code-agent "объясни, чем struct отличается от class в Swift"
```

Запрос с файлом:

```bash
code-agent --file Sources/App.swift "найди ошибки"
```

Только часть файла:

```bash
code-agent --file Sources/App.swift --range 40:120 "проверь этот участок"
```

Большие файлы требуют подтверждения. Отправить без подтверждения:

```bash
code-agent --file big_file.py --force-file "проверь файл"
```

Изменить лимит большого файла:

```bash
code-agent --file big_file.py --max-file-bytes 200000 "проверь файл"
```

## Вывод в терминале

- ввод пользователя зелёный;
- ответы агента голубые;
- код выводится в отдельной рамке с номерами строк;
- пока идёт запрос к API, показывается `Думаю...`.

Отключить цвета:

```bash
NO_COLOR=1 code-agent
```

## Настройки

По умолчанию:

```text
API URL:       https://api.deepseek.com/chat/completions
Модель:        deepseek-v4-flash
Temperature:   0.2
История:       20 сообщений
Лимит файла:   120 KB
```

Можно изменить через переменные окружения:

```bash
export CODE_AGENT_MODEL="deepseek-v4-flash"
export CODE_AGENT_TEMPERATURE="0.2"
export CODE_AGENT_MAX_HISTORY="20"
export CODE_AGENT_MAX_FILE_BYTES="122880"
```

Проверить текущие настройки внутри чата:

```text
/status
```

## Частые проблемы

`code-agent: command not found`

```bash
source .venv/bin/activate
which code-agent
```

`Не задан DEEPSEEK_API_KEY`

```bash
export DEEPSEEK_API_KEY="ваш_реальный_deepseek_ключ"
```

Ошибка установки:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

Файл слишком большой:

```bash
code-agent --file path/to/file.py --range 1:120 "проверь участок"
code-agent --file path/to/file.py --force-file "проверь весь файл"
```

## Разработка

```bash
python -m pip install -e .
python -m compileall code_agent_cli
```

