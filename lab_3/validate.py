import csv
import re
from pathlib import Path

from checksum import calculate_checksum, serialize_result


VARIANT = 49
CSV_PATH = Path(__file__).with_name("variant49.csv")

PATTERNS = {
    "email": re.compile(
        r"^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*@"
        r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.[A-Za-z]{2,}$"
    ),
    "http_status_message": re.compile(
        r"^[1-5][0-9]{2} [A-Za-z]+(?: [A-Za-z]+)*$"
    ),
    "snils": re.compile(r"^[0-9]{11}$"),
    "passport": re.compile(r"^[0-9]{2} [0-9]{2} [0-9]{6}$"),
    "ip_v4": re.compile(
        r"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$"
    ),
    "longitude": re.compile(
        r"^-?(?:180(?:\.0+)?|(?:1[0-7][0-9]|[1-9]?[0-9])(?:\.[0-9]+)?)$"
    ),
    "hex_color": re.compile(r"^#[0-9A-Fa-f]{6}$"),
    "isbn": re.compile(
        r"^(?:[0-9]-[0-9]{5}-[0-9]{3}-[0-9X]|"
        r"[0-9]{3}-[0-9]-[0-9]{5}-[0-9]{3}-[0-9])$"
    ),
    "locale_code": re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$"),
    "time": re.compile(
        r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{6}$"
    ),
}


def validate_file():
    invalid_rows = []
    invalid_counts = {column: 0 for column in PATTERNS}

    with CSV_PATH.open("r", encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        if reader.fieldnames != list(PATTERNS):
            raise ValueError("Неожиданные заголовки CSV")

        for row_number, row in enumerate(reader):
            row_is_invalid = False

            for column, pattern in PATTERNS.items():
                if not pattern.fullmatch(row[column]):
                    invalid_counts[column] += 1
                    row_is_invalid = True

            if row_is_invalid:
                invalid_rows.append(row_number)

    return invalid_rows, invalid_counts


def main():
    invalid_rows, invalid_counts = validate_file()

    print("Результаты проверки:")

    for column, count in invalid_counts.items():
        print(f"{column}: {count}")

    if any(count != 100 for count in invalid_counts.values()):
        raise SystemExit("Количество ошибок не совпадает с ожидаемым")

    if len(invalid_rows) != 1000:
        raise SystemExit("Количество невалидных строк не совпадает")

    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)

    print("\nВсего невалидных строк:", len(invalid_rows))
    print("Контрольная сумма:", checksum)
    print("Результат записан в result.json")


if __name__ == "__main__":
    main()
