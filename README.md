# CodeAgentCLI

Standalone Python CLI-ассистент для кодинга прямо в терминале.

CodeAgentCLI позволяет общаться с агентом из shell, задавать одноразовые вопросы,
прикладывать файлы с кодом и получать читаемый терминальный вывод: цветные ответы,
простую анимацию загрузки и выделенные блоки кода.

## Требования

- Python 3.9 или новее
- Git
- API-ключ DeepSeek

## Установка из GitHub

Склонируйте репозиторий:

```bash
git clone git@github.com:ipv02/CodeAgentCLI.git
cd CodeAgentCLI
```

Если у вас не настроены SSH-ключи для GitHub, используйте HTTPS:

```bash
git clone https://github.com/ipv02/CodeAgentCLI.git
cd CodeAgentCLI
```

Создайте виртуальное окружение:

```bash
python3 -m venv .venv
```

Активируйте его:

```bash
source .venv/bin/activate
```

Установите CLI:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

После установки команда `code-agent` будет доступна, пока активно виртуальное
окружение.

## API-ключ

CodeAgentCLI читает ключ из переменной окружения `DEEPSEEK_API_KEY`.

Быстрый временный вариант для текущего терминала:

```bash
export DEEPSEEK_API_KEY="ваш_реальный_deepseek_ключ"
```

Такой вариант работает только до закрытия текущего окна терминала.

Чтобы не вводить ключ каждый раз, добавьте его в `~/.zshrc`:

```bash
echo 'export DEEPSEEK_API_KEY="ваш_реальный_deepseek_ключ"' >> ~/.zshrc
source ~/.zshrc
```

Если вы используете `bash`, добавьте ключ в `~/.bashrc` или `~/.bash_profile`:

```bash
echo 'export DEEPSEEK_API_KEY="ваш_реальный_deepseek_ключ"' >> ~/.bashrc
source ~/.bashrc
```

Не добавляйте реальный API-ключ в файлы проекта и не коммитьте его в Git.
Храните ключ в настройках shell или в другом приватном хранилище секретов.

## Первый запуск

Из папки проекта выполните:

```bash
source .venv/bin/activate
code-agent
```

Вы должны увидеть:

```text
Code Agent CLI
Команды: /help, /reset, /exit
>
```

Теперь можно написать вопрос и нажать Enter:

```text
> объясни, чем struct отличается от class в Swift
```

Пока агент думает, CLI показывает простую анимацию:

```text
Думаю...
```

Ответ будет выведен с цветным форматированием, если терминал поддерживает ANSI
цвета.

## Обычный запуск каждый день

Если вы открыли новый терминал:

```bash
cd CodeAgentCLI
source .venv/bin/activate
code-agent
```

Не нужно каждый раз заново создавать `.venv` или переустанавливать пакет.

Эти команды обычно выполняются один раз при первой установке:

```bash
python3 -m venv .venv
python -m pip install .
```

А эту команду нужно выполнять в каждом новом терминале, если вы не сделали
shortcut:

```bash
source .venv/bin/activate
```

## Удобный shortcut

Если хотите запускать `code-agent` из любой папки без ручной активации `.venv`,
добавьте функцию в `~/.zshrc`.

Замените `/path/to/CodeAgentCLI` на реальный путь к папке, куда вы скачали проект:

```bash
code-agent() {
  /path/to/CodeAgentCLI/.venv/bin/code-agent "$@"
}
```

Примените настройки:

```bash
source ~/.zshrc
```

После этого можно запускать агента из любой папки:

```bash
code-agent
```

## Команды в интерактивном режиме

Внутри чата доступны команды:

```text
/help   показать помощь
/reset  очистить историю текущей сессии
/exit   выйти из CLI
/quit   тоже выйти из CLI
```

История диалога хранится только в памяти текущего процесса. Если выйти из CLI,
история исчезнет.

## Одноразовый запрос

Можно задать один вопрос без входа в интерактивный режим:

```bash
code-agent "объясни, чем struct отличается от class в Swift"
```

CLI отправит запрос, напечатает ответ и завершит работу.

## Запрос с файлом

Чтобы приложить файл с кодом, используйте `--file` или `-f`:

```bash
code-agent --file Sources/App.swift "найди ошибки"
```

Короткий вариант:

```bash
code-agent -f Sources/App.swift "объясни этот код"
```

CLI прочитает файл как UTF-8 и отправит его содержимое вместе с вашим вопросом.

## Оформление вывода

В обычном терминале CodeAgentCLI использует ANSI-цвета:

- ввод пользователя подсвечивается зелёным;
- ответы агента выводятся голубым;
- блоки кода показываются отдельно в рамке;
- в блоках кода есть номера строк и простая подсветка синтаксиса;
- пока идёт запрос к API, показывается лоадер `Думаю...`.

Чтобы отключить цвета, запустите CLI с `NO_COLOR`:

```bash
NO_COLOR=1 code-agent
```

## Настройки агента

Текущие настройки находятся в `code_agent_cli/agent.py`:

```text
API URL:              https://api.deepseek.com/chat/completions
Модель:               deepseek-v4-flash
Temperature:          0.2
Лимит истории:        20 сообщений
```

System prompt настраивает агента как помощника для написания, объяснения,
улучшения и отладки кода. Агент должен отвечать практично и кратко, а если
информации недостаточно, задавать уточняющий вопрос.

Сейчас эти настройки меняются через исходный код. Отдельных CLI-флагов для
модели, температуры и system prompt пока нет.

## Частые проблемы

### `code-agent: command not found`

Проверьте, что виртуальное окружение активировано:

```bash
source .venv/bin/activate
```

Затем проверьте путь до команды:

```bash
which code-agent
```

Команда должна указывать на `.venv/bin/code-agent`.

### `Не задан DEEPSEEK_API_KEY`

Задайте API-ключ:

```bash
export DEEPSEEK_API_KEY="ваш_реальный_deepseek_ключ"
```

Чтобы не вводить его каждый раз, добавьте эту строку в `~/.zshrc` или
`~/.bashrc`.

### Ошибка при установке

Обновите инструменты упаковки и повторите установку:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

### Не удаётся прочитать файл

Опция `--file` ожидает путь к читаемому текстовому файлу в UTF-8:

```bash
code-agent --file path/to/file.py "проверь этот файл"
```

## Разработка

Для локальной разработки можно установить пакет в editable-режиме:

```bash
python -m pip install -e .
```

Если editable-установка падает из-за старых версий `pip` или `setuptools`,
обновите инструменты:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Быстрая проверка синтаксиса:

```bash
python -m compileall code_agent_cli
```

