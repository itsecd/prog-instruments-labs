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
make install-hooks  # настройка локальных Git hooks
```

После `make install-hooks` хук `pre-commit` запускает линтер и проверку форматирования,
а `pre-push` запускает все проверки. Файлы хуков хранятся в `.githooks/`.

Конфигурации Ruff и Pyright находятся соответственно в `pyproject.toml`
и `pyrightconfig.json`. Выбран режим Pyright `strict`.
