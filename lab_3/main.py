import csv
import json
import re
from typing import List

from checksum import calculate_checksum

PATTERNS = {
    "telephone": r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$",
    "height": r"^[12]\.\d{2}$",
    "snils": r"^\d{11}$",
    "identifier": r"^\d{2}-\d{2}/\d{2}$",
    "occupation": r"^[А-Яа-яЁёA-Za-z\s-]+$",
    "longitude": r"^-?(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?$|^-?180(?:\.0+)?$",
    "blood_type": r"^(?:A|B|AB|O)[+\u2212]$",
    "issn": r"^\d{4}-\d{4}$",
    "locale_code": r"^[a-z]{2}(-[a-z]{2})?$",
    "date": r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$",
}

def validate_file(filepath: str) -> List[int]:
    """
    Читает CSV-файл и возвращает список номеров строк (с нуля, без учёта заголовка),
    в которых хотя бы одна ячейка не прошла валидацию.
    """
    invalid_rows = []

    with open(filepath, mode="r", encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row_index, row in enumerate(reader):
            is_invalid = False

            for column, value in row.items():
                pattern = PATTERNS.get(column)
                if pattern is None:
                    continue

                if not re.fullmatch(pattern, value):
                    is_invalid = True
                    break

            if is_invalid:
                invalid_rows.append(row_index)

    return invalid_rows

def main() -> None:
    variant = 60
    csv_path = "60.csv"

    invalid_rows = validate_file(csv_path)
    checksum = calculate_checksum(invalid_rows)

    print(f"Контрольная сумма: {checksum}")

if __name__ == "__main__":
    main()