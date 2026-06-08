# CodeAgentCLI

Standalone Python CLI-ассистент для кодинга. Агентская логика и system prompt скопированы из `LLMRequest/Service/MyAgentService.swift`, но проект не зависит от `LLMRequest`.

## Установка для разработки

```bash
cd /Users/ipv/Desktop/Develop/CodeAgentCLI
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Ключ API

```bash
export DEEPSEEK_API_KEY="ваш_ключ"
```

## Интерактивный режим

```bash
code-agent
```

Команды внутри чата:

```text
/help
/reset
/exit
```

## Одноразовый запрос

```bash
code-agent "объясни, чем struct отличается от class в Swift"
```

## Запрос с файлом

```bash
code-agent --file Sources/App.swift "найди ошибки"
```
