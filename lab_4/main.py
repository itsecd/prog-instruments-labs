import argparse
from config import DATA_PATH, VARIANT, FIRST_DATA_ROW_INDEX
from src.reader import read_csv
# main.py
# уточните имя функции в докстрингах модуля
from checksum import calculate_checksum, serialize_result
from src.validators import validate_row
from pathlib import Path


def process_csv(filepath) -> list[int]:
    invalid_rows = []
    for line_num, row in enumerate(
        read_csv(filepath), start=FIRST_DATA_ROW_INDEX
    ):
        if not validate_row(row):
            invalid_rows.append(line_num)
    return invalid_rows


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Lab 3: CSV validation")
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH,
        help="путь к CSV-файлу (по умолчанию data/36.csv)",
    )
    parser.add_argument(
        "--variant", type=int, default=VARIANT,
        help="номер варианта (по умолчанию 36)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = process_csv(args.data)
    checksum = calculate_checksum(invalid)
    print(f"Невалидных строк: {len(invalid)}")
    print(f"Контрольная сумма: {checksum}")
    serialize_result(args.variant, checksum)


if __name__ == "__main__":
    main()
