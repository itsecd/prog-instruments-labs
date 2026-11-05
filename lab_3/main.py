"""Лабораторная работа №3, вариант 74: валидация csv-файла регулярными выражениями.

Программа читает csv-файл, проверяет каждую ячейку регулярным выражением
для её столбца, находит номера невалидных строк, считает контрольную сумму
и записывает результат в result.json.
"""

import csv
import re

from checksum import calculate_checksum, serialize_result

VARIANT = 74
CSV_PATH = "74.csv"

# Число от 0 до 255 — одна часть IP-адреса.
OCTET = r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"

# Для каждого столбца файла — своё регулярное выражение.
PATTERNS = {
    # +7-(963)-651-44-40
    "telephone": r"\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}",
    # 305 Use Proxy, 226 IM Used
    "http_status_message": r"[1-5]\d{2}( [A-Za-z]+)+",
    # 59343138004 — 11 цифр
    "snils": r"\d{11}",
    # 87-01/65
    "identifier": r"\d{2}-\d{2}/\d{2}",
    # 55.246.134.149 — четыре числа от 0 до 255
    "ip_v4": "(" + OCTET + r"\.){3}" + OCTET,
    # 48.143635, -151.295931 — от -180 до 180
    "longitude": r"-?(180(\.0+)?|(1[0-7]\d|[1-9]?\d)\.\d+)",
    # A+, O−, AB+ (отрицательный резус записан символом U+2212)
    "blood_type": r"(A|B|AB|O)[+−]",
    # 1-88425-228-3 (10 цифр), 351-1-91934-017-1 (13 цифр)
    "isbn": r"(\d{3}-)?\d-\d{5}-\d{3}-\d",
    # no, ro-mo
    "locale_code": r"[a-z]{2}(-[a-z]{2})?",
    # 2015-09-04 — месяц 01–12, день 01–31
    "date": r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])",
}


def read_csv(path: str) -> tuple[list[str], list[list[str]]]:
    """Читает csv-файл и возвращает заголовки столбцов и строки с данными."""
    with open(path, encoding="utf-16", newline="") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader)
        rows = list(reader)
    return header, rows


def is_valid_row(header: list[str], row: list[str]) -> bool:
    """Проверяет, что каждая ячейка строки подходит под шаблон своего столбца."""
    for index, column in enumerate(header):
        if re.fullmatch(PATTERNS[column], row[index]) is None:
            return False
    return True


def find_invalid_rows(header: list[str], rows: list[list[str]]) -> list[int]:
    """Возвращает номера невалидных строк (первая строка данных имеет номер 0)."""
    invalid_rows = []
    for number, row in enumerate(rows):
        if not is_valid_row(header, row):
            invalid_rows.append(number)
    return invalid_rows


def main() -> None:
    header, rows = read_csv(CSV_PATH)
    invalid_rows = find_invalid_rows(header, rows)
    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)
    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")


if __name__ == "__main__":
    main()
