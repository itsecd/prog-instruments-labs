"""Валидация csv-файла варианта и запись контрольной суммы в result.json."""

from pathlib import Path

from checksum import calculate_checksum, serialize_result
from validator import find_invalid_row_numbers, read_rows

VARIANT = 18
CSV_PATH = Path(__file__).parent / f"{VARIANT}.csv"


def main() -> None:
    """Основная функция программы."""
    rows = read_rows(CSV_PATH)
    invalid_row_numbers = find_invalid_row_numbers(rows)
    checksum = calculate_checksum(invalid_row_numbers)
    serialize_result(VARIANT, checksum)

    print(f"Всего строк: {len(rows)}")
    print(f"Невалидных строк: {len(invalid_row_numbers)}")
    print(f"Контрольная сумма: {checksum}")


if __name__ == "__main__":
    main()
