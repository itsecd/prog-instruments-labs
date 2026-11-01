"""Чтение CSV-файла и поиск строк с невалидными значениями."""

import csv
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

CSV_ENCODING = "utf-16"
CSV_DELIMITER = ";"


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Читает CSV-файл и возвращает заголовок и строки данных."""
    with path.open(encoding=CSV_ENCODING, newline="") as file:
        rows = list(csv.reader(file, delimiter=CSV_DELIMITER))
    return rows[0], rows[1:]


def is_valid_row(
    header: Iterable[str],
    row: Iterable[str],
    patterns: Mapping[str, re.Pattern[str]],
) -> bool:
    """Проверяет, что каждое значение строки соответствует шаблону столбца."""
    return all(
        patterns[column].fullmatch(value) is not None
        for column, value in zip(header, row, strict=True)
    )


def find_invalid_rows(
    header: list[str],
    rows: list[list[str]],
    patterns: Mapping[str, re.Pattern[str]],
) -> list[int]:
    """Возвращает номера строк с ошибками."""
    unknown = set(header) - patterns.keys()
    if unknown:
        raise ValueError(f"Нет шаблонов для столбцов: {sorted(unknown)}")
    return [
        number
        for number, row in enumerate(rows)
        if not is_valid_row(header, row, patterns)
    ]
