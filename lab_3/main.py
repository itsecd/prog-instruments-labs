import csv
import json
import re
from typing import List

from checksum import calculate_checksum

PATTERNS = {}

def validate_file(filepath: str) -> List[int]:
    """
    Читает CSV-файл и возвращает список номеров строк (с нуля, без учёта заголовка),
    в которых хотя бы одна ячейка не прошла валидацию.
    """
    invalid_rows = []

    with open(filepath, mode="r", encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row_index, row in enumerate(reader):
            pass

    return invalid_rows

def main() -> None:
    variant = 60
    csv_path = "60.csv"

    invalid_rows = validate_file(csv_path)
    checksum = calculate_checksum(invalid_rows)

    print(f"Контрольная сумма: {checksum}")

    serialize_result(variant, checksum)

if __name__ == "__main__":
    main()