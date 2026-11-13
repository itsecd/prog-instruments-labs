"""
Лабораторная работа № 3 «Regular expressions».
Вариант 16.

Проверяет 10 столбцов CSV-файла регулярными выражениями,
находит строки с ошибками и вычисляет контрольную сумму.
"""

import csv
import datetime
import json
import re
from pathlib import Path

from checksum import calculate_checksum, serialize_result

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "16.csv"
REPORT_PATH = BASE_DIR / "validation_report.json"

VARIANT = 16

# По одному регулярному выражению для каждого столбца.
PATTERNS = {
    "telephone": re.compile(r"\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}"),
    "height": re.compile(r"(?:0|1|2)\.\d{2}"),
    "inn": re.compile(r"\d{12}"),
    "identifier": re.compile(r"\d{2}-\d{2}/\d{2}"),
    "occupation": re.compile(
        r"[A-Za-zА-Яа-яЁё]+(?:[ -][A-Za-zА-Яа-яЁё]+)*"
    ),
    "latitude": re.compile(r"-?\d{1,2}\.\d+"),
    "blood_type": re.compile(r"(?:A|B|AB|O)[+−]"),
    "issn": re.compile(r"\d{4}-\d{4}"),
    "uuid": re.compile(
        r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    ),
    "date": re.compile(r"\d{4}-\d{2}-\d{2}"),
}


def is_valid(field: str, value: str) -> bool:
    """Проверяет формат значения и дополнительные ограничения поля."""
    if not PATTERNS[field].fullmatch(value):
        return False

    if field == "height":
        height = float(value)
        return 0 < height < 3

    if field == "latitude":
        latitude = float(value)
        return -90 <= latitude <= 90

    if field == "date":
        try:
            datetime.date.fromisoformat(value)
        except ValueError:
            return False

    return True


def read_csv() -> tuple[list[str], list[dict[str, str]]]:
    """Читает исходный CSV в кодировке UTF-16 с разделителем ';'."""
    with CSV_PATH.open("r", encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        if reader.fieldnames != list(PATTERNS):
            raise ValueError(
                "Столбцы CSV не совпадают с ожидаемыми: "
                + ", ".join(reader.fieldnames or [])
            )
        rows = list(reader)

    if len(rows) != 10_000:
        raise ValueError(f"Ожидалось 10000 строк, получено {len(rows)}.")

    return list(PATTERNS), rows


def validate_rows(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[dict[str, list[int]], list[int]]:
    """
    Возвращает ошибки по каждому полю и общий список ошибочных строк.

    Номер строки данных начинается с 0: первая строка после заголовка
    имеет номер 0, как указано в checksum.py.
    """
    invalid_by_field = {field: [] for field in headers}
    invalid_rows = set()

    for row_number, row in enumerate(rows):
        for field in headers:
            value = row[field]
            if not is_valid(field, value):
                invalid_by_field[field].append(row_number)
                invalid_rows.add(row_number)

    return invalid_by_field, sorted(invalid_rows)


def main() -> None:
    """Выполняет проверку и сохраняет результат для GitHub Actions."""
    headers, rows = read_csv()
    invalid_by_field, invalid_rows = validate_rows(headers, rows)
    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)

    # Подробный локальный отчёт для демонстрации преподавателю.
    report = {
        "variant": VARIANT,
        "rows_checked": len(rows),
        "columns_checked": len(headers),
        "invalid_count_by_column": {
            field: len(numbers)
            for field, numbers in invalid_by_field.items()
        },
        "unique_invalid_rows_count": len(invalid_rows),
        "invalid_rows_zero_based": invalid_rows,
        "checksum": checksum,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"Вариант: {VARIANT}")
    print(f"Проверено строк: {len(rows)}")
    print(f"Проверено столбцов: {len(headers)}")
    print("Невалидных значений по столбцам:")
    for field, numbers in invalid_by_field.items():
        print(f"  {field}: {len(numbers)}")
    print(f"Уникальных строк с ошибками: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")
    print("Результат сохранён в result.json")
    print(f"Подробный отчёт сохранён в {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
