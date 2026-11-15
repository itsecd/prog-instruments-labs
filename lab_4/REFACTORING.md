# Рефакторинг лабораторной №3

## Исходные проблемы (code smells)

1. Хардкод пути `"data/36.csv"` и варианта `36` в `main.py`.
2. `main.py` смешивал чтение CSV, валидацию и сериализацию.
3. `validate_row` возвращал `bool`, но `find_invalid_fields`-логика
   дублировалась в `diagnose.py`.
4. `KeyError` при неизвестном поле — неясно, ошибка это или часть логики.
5. Нет тестов.
6. Магическое `start=0` без объяснения.
7. `diagnose.py` дублировал обход полей.

## Что сделано (по коммитам)

1. `config.py` — вынесены константы (`DATA_PATH`, `VARIANT`, `CSV_*`).
2. `reader.py` — чтение CSV отделено от валидации.
3. `argparse` — `--data` и `--variant` вместо хардкода.
4. `validators.py` — `validate_row` (`bool`) + `find_invalid_fields` (`list[str]`).
5. `diagnose.py` — переиспользует `find_invalid_fields`, без дублирования.
6. `tests/` — `pytest` на валидаторы и на полный прогон.
7. `flake8` — 0 ошибок, код по PEP 8.

## Поведение сохранено

- `py main.py` → **1000** невалидных строк
- checksum: **`ceea41d057c2bfc5682c8886e74dfc24`**
- `pytest tests/` → все тесты проходят
