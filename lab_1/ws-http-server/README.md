# websockets-http-server

Простой HTTP-сервер на `aiohttp` и WebSocket-сервер на `websockets`.
Браузерная страница показывает текущее локальное время сервера, обновляемое каждую секунду.

## Запуск

Требуется Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Откройте http://localhost:7878. WebSocket использует `ws://localhost:8020`.
Для остановки сервера нажмите Ctrl+C.

## Инструменты разработки

```bash
python -m pip install -r requirements-dev.txt
make lint           # Ruff: статический анализ
make format         # Ruff: автоматическое форматирование
make format-check   # проверка форматирования без изменений
make typecheck      # Pyright: строгая проверка типов
make test           # интеграционные тесты
make check          # все проверки
make precommit      # установка pre-commit и pre-push хуков
```

Фреймворк pre-commit использует `.pre-commit-config.yaml` с разделением на стадии:

- `pre-commit` (`git commit`) — Ruff lint и проверка форматирования;
- `pre-push` (`git push`) — Pyright и pytest.

Хуки проверяют только проект `lab_1/ws-http-server` и запускают команды Makefile.
Команда `make precommit` устанавливает оба хука в Git-репозиторий; она также
удаляет прежнюю локальную настройку `core.hooksPath`, если она равна `.githooks`.
Ветка без данной конфигурации не блокируется установленными хуками.

Для ручной проверки обеих стадий из корня Git-репозитория (с активированной `.venv`):

```bash
pre-commit run --config lab_1/ws-http-server/.pre-commit-config.yaml --all-files --hook-stage pre-commit
pre-commit run --config lab_1/ws-http-server/.pre-commit-config.yaml --all-files --hook-stage pre-push
```

При изменении файлов проекта выполните обычные `git add`, `git commit` и `git push`,
чтобы увидеть автоматический запуск соответствующих проверок.

Конфигурации Ruff и Pyright находятся соответственно в `pyproject.toml`
и `pyrightconfig.json`. Выбран режим Pyright `strict`.
