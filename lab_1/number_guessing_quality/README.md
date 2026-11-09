# Лабораторная работа №1

Исходный проект: [Number Guessing Game](https://github.com/AsifTLT/Number-Guessing-Game) (MIT, лицензия в `LICENSE`).

В игре вынесено сравнение числа в отдельную функцию, добавлены тесты. Для проверки кода используются Ruff и Pyright, для запуска команд — Makefile. Перед коммитом запускаются проверки Ruff, перед отправкой — Pyright и тесты. GitHub Actions проверяет код при push и создании PR.

## Запуск и проверка

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
make fulltest
```

Для установки хуков из каталога проекта:

```powershell
make precommit
```

Если `make` не установлен, команды проверки приведены в `WINDOWS_INSTRUCTIONS.md` в корне архива.
