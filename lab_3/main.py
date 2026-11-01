"""Валидация CSV-файла регулярными выражениями."""

import argparse
from pathlib import Path

from checksum import calculate_checksum, serialize_result
from patterns import PATTERNS
from validator import find_invalid_rows, read_csv

VARIANT = 19
DEFAULT_CSV_PATH = Path(__file__).parent / f"{VARIANT}.csv"


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Поиск невалидных строк CSV и подсчёт контрольной суммы"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Путь к CSV-файлу (по умолчанию {DEFAULT_CSV_PATH.name})",
    )
    return parser.parse_args()


def main() -> None:
    """Находит невалидные строки и записывает результат в result.json."""
    args = parse_args()
    header, rows = read_csv(args.csv_path)
    invalid_rows = find_invalid_rows(header, rows, PATTERNS)
    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)
    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")


if __name__ == "__main__":
    main()
