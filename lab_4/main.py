from config import DATA_PATH, VARIANT
from reader import read_csv
# main.py
# уточните имя функции в докстрингах модуля
from checksum import calculate_checksum, serialize_result
from src.validators import validate_row


def process_csv(filepath) -> list[int]:
    invalid_rows = []
    for line_num, row in enumerate(
        read_csv(filepath), start=FIRST_DATA_ROW_INDEX
    ):
        if not validate_row(row):
            invalid_rows.append(line_num)
    return invalid_rows


if __name__ == "__main__":
    invalid = process_csv(DATA_PATH)
    checksum = calculate_checksum(invalid)  # см. докстринги checksum.py
    print(f"Невалидных строк: {len(invalid)}")
    print(f"Контрольная сумма: {checksum}")
    serialize_result(VARIANT, checksum=checksum)
