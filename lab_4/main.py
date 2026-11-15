from config import DATA_PATH, VARIANT

# main.py
import csv
# уточните имя функции в докстрингах модуля
from checksum import calculate_checksum, serialize_result
from src.validators import validate_row


def process_csv(filepath: str) -> list[int]:
    invalid_rows = []
    with open(filepath, encoding="utf-16", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for line_num, row in enumerate(reader, start=0):
            if not validate_row(row):
                invalid_rows.append(line_num)   # ← list[int], 1-based
    return invalid_rows


if __name__ == "__main__":
    invalid = process_csv(DATA_PATH)
    checksum = calculate_checksum(invalid)  # см. докстринги checksum.py
    print(f"Невалидных строк: {len(invalid)}")
    print(f"Контрольная сумма: {checksum}")
    serialize_result(VARIANT, checksum=checksum)
