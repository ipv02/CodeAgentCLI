# Запуск CodeAgentCLI LLM-сервиса на VPS

Короткий сценарий для запуска приватного HTTP-сервиса локальной LLM на VPS.

## 1. Подготовить VPS

Подключитесь к серверу:

```bash
ssh root@VPS_IP
```

Установите базовые пакеты:

```bash
apt update
apt install -y git curl python3 python3-venv python3-pip
```

Перейдите в проект:

```bash
cd ~/CodeAgentCLI
```

Если проекта еще нет на VPS, сначала склонируйте репозиторий:

```bash
git clone REPOSITORY_URL CodeAgentCLI
cd CodeAgentCLI
```

## 2. Установить агент в виртуальное окружение

Создайте окружение и установите проект:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install .
```

Если VPS использует Python 3.12, а проект требует Python 3.14+, для учебного
запуска можно временно поменять в `pyproject.toml`:

```toml
requires-python = ">=3.12"
```

Это локальная правка только на VPS. Не коммитьте ее, если основная версия
проекта должна оставаться Python 3.14+.

## 3. Установить и запустить Ollama

Установите Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Обычно после установки Ollama поднимается как системный сервис. Проверьте:

```bash
systemctl status ollama
```

Если сервис не запущен, поднимите его:

```bash
systemctl start ollama
```

Проверьте, что локальный API Ollama отвечает:

```bash
curl http://127.0.0.1:11434/api/tags
```

Скачайте локальную модель:

```bash
ollama pull llama3.2:3b
```

Проверить модель можно так:

```bash
ollama run llama3.2:3b
```

Выйти из интерактивного режима Ollama:

```bash
/bye
```

Если `systemctl` недоступен, можно запустить Ollama вручную в отдельном
терминале:

```bash
ollama serve
```

Остановить Ollama как системный сервис:

```bash
systemctl stop ollama
```

Снова поднять:

```bash
systemctl start ollama
```

Перезапустить:

```bash
systemctl restart ollama
```

## 4. Запустить LLM-сервис

Из папки проекта запустите агент как HTTP-сервис:

```bash
CODE_AGENT_LLM_SERVICE_USERNAME='admin' \
CODE_AGENT_LLM_SERVICE_PASSWORD='YOUR_STRONG_PASSWORD' \
.venv/bin/agent --llm-service --llm-service-host 0.0.0.0 --llm-service-port 8080
```

`0.0.0.0` означает, что сервис слушает внешние подключения. Открывайте чат по
адресу:

```text
http://VPS_IP:8080/chat
```

Логин:

```text
admin
```

Пароль:

```text
YOUR_STRONG_PASSWORD
```

Если нужно запустить агента в фоне и освободить терминал, добавьте `&` в конце:

```bash
CODE_AGENT_LLM_SERVICE_USERNAME='admin' \
CODE_AGENT_LLM_SERVICE_PASSWORD='YOUR_STRONG_PASSWORD' \
.venv/bin/agent --llm-service --llm-service-host 0.0.0.0 --llm-service-port 8080 &
```

Проверить, что агент слушает внешний порт:

```bash
ss -ltnp | grep 8080
```

В корректном варианте в выводе должен быть адрес:

```text
0.0.0.0:8080
```

## 5. Проверить доступ из сети

Откройте в браузере на своем Mac или телефоне:

```text
http://VPS_IP:8080/chat
```

Если страница не открывается, проверьте порт на VPS:

```bash
ss -ltnp | grep 8080
```

Проверьте firewall на сервере:

```bash
ufw status
ufw allow 8080/tcp
```

Также проверьте firewall/security group в панели провайдера VPS.

Если на Mac адрес открывается, а на телефоне через мобильный интернет нет,
проверьте телефон через Wi-Fi. Если через Wi-Fi открывается, значит агент,
VPS и порт работают, а проблема в мобильной сети. Частые причины:

- мобильный оператор блокирует или нестабильно маршрутизирует нестандартный
  порт `8080`;
- на телефоне включен VPN, iCloud Private Relay или защита трафика;
- браузер телефона пытается открыть `https://` вместо `http://`;
- у оператора есть ограничения на прямой доступ к таким портам.

Для более стабильного публичного доступа лучше использовать обычные порты
`80`/`443`, домен и HTTPS через reverse proxy. Для быстрой проверки можно
временно запустить сервис на порту `80`:

```bash
CODE_AGENT_LLM_SERVICE_USERNAME='admin' \
CODE_AGENT_LLM_SERVICE_PASSWORD='YOUR_STRONG_PASSWORD' \
.venv/bin/agent --llm-service --llm-service-host 0.0.0.0 --llm-service-port 80
```

Тогда адрес будет:

```text
http://VPS_IP/chat
```

## 6. Проверочные вопросы про кино

После авторизации задайте в чате:

```text
Посоветуй 3 фильма в жанре нуар и кратко объясни, чем они отличаются.
```

```text
Сравни актерскую манеру Аль Пачино и Роберта Де Ниро на примерах фильмов.
```

```text
Объясни простыми словами, чем режиссура Кристофера Нолана отличается от режиссуры Дени Вильнева.
```

После каждого ответа на странице должны отображаться ограничения и метаданные:

- rate limit;
- max context;
- max output tokens;
- доступные Ollama usage metrics, если модель их вернула.

## 7. Остановить сервис

Если сервис запущен в текущем терминале, остановите его:

```bash
Ctrl+C
```

Если процесс запущен в фоне, найдите его:

```bash
lsof -i :8080
```

Затем завершите по PID:

```bash
kill PID
```

Если процесс не остановился:

```bash
kill -9 PID
```

Можно остановить все фоновые LLM-сервисы агента одной командой:

```bash
pkill -f "agent --llm-service"
```

После остановки проверьте, что порт освободился:

```bash
ss -ltnp | grep 8080
```

Если вывода нет, сервис выключен.

## 8. Важные нюансы безопасности

Если сервис запущен с `--llm-service-host 0.0.0.0` и порт `8080` открыт наружу,
страницу сможет открыть любой, кто знает IP и порт. Сам чат защищен логином и
паролем, но для реального использования лучше:

- использовать сильный уникальный пароль;
- не коммитить реальные пароли, токены и API keys;
- открыть доступ только через VPN, Tailscale или firewall allowlist;
- добавить HTTPS через reverse proxy;
- не публиковать API key, если используете Bearer auth.
