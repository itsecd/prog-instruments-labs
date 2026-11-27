"""Валидация CSV с помощью регулярных выражений (вариант 51)."""

import argparse
import csv
import re
from pathlib import Path

from checksum import calculate_checksum, serialize_result


# Выражения можно корректировать независимо от механизма обработки CSV.
PATTERNS = {
    "email": r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*@[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+",
    "height": r"[012]\.[0-9]{2}",
    "snils": r"[0-9]{11}",
    "passport": r"[0-9]{2} [0-9]{2} [0-9]{6}",
    "occupation": r"[A-Za-zА-Яа-яЁё]+(?:[ -][A-Za-zА-Яа-яЁё]+)*",
    "longitude": (
        r"-?(?:(?:1[0-7][0-9]|[0-9]{1,2})(?:\.[0-9]+)?|180(?:\.0+)?)"
    ),
    "hex_color": r"#[0-9A-Fa-f]{6}",
    "issn": r"[0-9]{4}-[0-9]{4}",
    "locale_code": r"[a-z]{2}(?:-[a-z]{2}(?:\.[a-z]{2})?)?",
    "time": r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]{1,6})?",
}

VARIANT = 51
BASE_DIR = Path(__file__).resolve().parent


def find_invalid_rows(csv_path: Path) -> tuple[list[int], dict[str, int]]:
    """Возвращает нулевые индексы невалидных строк и счётчики ошибок полей."""
    validators = {
        column: re.compile(pattern) for column, pattern in PATTERNS.items()
    }
    invalid_rows = []
    counts = dict.fromkeys(PATTERNS, 0)

    with csv_path.open(encoding="utf-16", newline="") as source:
        reader = csv.DictReader(source, delimiter=";")
        if reader.fieldnames is None or set(reader.fieldnames) != set(PATTERNS):
            raise ValueError("Заголовки CSV не соответствуют ожидаемым полям")

        for row_number, row in enumerate(reader):
            invalid = False
            for column, validator in validators.items():
                value = row[column]
                if value is None or validator.fullmatch(value) is None:
                    counts[column] += 1
                    invalid = True
            if invalid:
                invalid_rows.append(row_number)

    return invalid_rows, counts


def main() -> None:
    """Валидирует CSV, вычисляет MD5 и сохраняет result.json."""
    parser = argparse.ArgumentParser(description="Regex validation, variant 51")
    parser.add_argument(
        "--csv",
        type=Path,
        default=BASE_DIR / "51.csv",
        help="Путь к CSV для проверки (по умолчанию lab_3/51.csv)",
    )
    arguments = parser.parse_args()

    invalid_rows, counts = find_invalid_rows(arguments.csv)
    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)

    for column, count in counts.items():
        print(f"{column}: {count}")
    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")
    print(f"Результат: {BASE_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
