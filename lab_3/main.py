# main.py
import csv
# уточните имя функции в докстрингах модуля
from checksum import calculate_checksum, serialize_result
from src.validators import validate_row


def process_csv(filepath: str) -> list[int]:
    """Читает CSV и возвращает номера невалидных строк (1-based)."""
    invalid_rows = []
    with open(filepath, encoding="utf-16") as f:
        reader = csv.DictReader(f, delimiter=";")
        for line_num, row in enumerate(reader, start=1):
            if validate_row(row):
                invalid_rows.append(line_num)
    return invalid_rows


if __name__ == "__main__":
    invalid = process_csv("data/36.csv")
    checksum = calculate_checksum(invalid)  # см. докстринги checksum.py
    print(f"Невалидных строк: {len(invalid)}")
    print(f"Контрольная сумма: {checksum}")
    serialize_result(36, checksum=checksum)
