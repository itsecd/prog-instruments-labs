"""Валидация строк csv-файла с помощью регулярных выражений."""

import csv
import re
from pathlib import Path

CSV_ENCODING = "utf-16"
CSV_DELIMITER = ";"

OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"

RAW_PATTERNS: dict[str, str] = {
    "telephone": r"\+7-\([0-9]{3}\)-[0-9]{3}-[0-9]{2}-[0-9]{2}",
    "http_status_message": r"[1-5][0-9]{2} [A-Za-z]+(?: [A-Za-z]+)*",
    "snils": r"[0-9]{11}",
    "identifier": r"[0-9]{2}-[0-9]{2}/[0-9]{2}",
    "ip_v4": rf"(?:{OCTET}\.){{3}}{OCTET}",
    "longitude": r"-?(?:180(?:\.0+)?|(?:1[0-7][0-9]|[1-9]?[0-9])(?:\.[0-9]+)?)",
    "blood_type": r"(?:A|B|AB|O)[+\u2212]",
    "isbn": r"(?:[0-9]{3}-)?[0-9]-[0-9]{5}-[0-9]{3}-[0-9]",
    "locale_code": r"[a-z]{2}(?:-[a-z]{2})?",
    "date": r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])",
}

PATTERNS: dict[str, re.Pattern[str]] = {
    column: re.compile(regex) for column, regex in RAW_PATTERNS.items()
}


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Читает csv-файл и возвращает список строк в виде словарей.

    Args:
        csv_path: Путь к csv-файлу

    Returns:
        Список строк данных (без строки заголовков)
    """
    with csv_path.open(encoding=CSV_ENCODING, newline="") as file:
        return [dict(row) for row in csv.DictReader(file, delimiter=CSV_DELIMITER)]


def is_valid_row(row: dict[str, str]) -> bool:
    """Проверяет, что все значения строки соответствуют своим шаблонам.

    Args:
        row: Строка данных, ключ - название колонки

    Returns:
        True, если каждое значение прошло валидацию, иначе False
    """
    return all(
        PATTERNS[column].fullmatch(value) is not None for column, value in row.items()
    )


def find_invalid_row_numbers(rows: list[dict[str, str]]) -> list[int]:
    """Находит номера строк, не прошедших валидацию.

    Нумерация начинается с 0 для первой строки данных (как требует checksum.py).

    Args:
        rows: Строки данных

    Returns:
        Список номеров невалидных строк
    """
    return [number for number, row in enumerate(rows) if not is_valid_row(row)]
