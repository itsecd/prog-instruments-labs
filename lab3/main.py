import csv
import re
from pathlib import Path

from checksum import calculate_checksum, serialize_result


PATTERNS = {
    "telephone":  re.compile(r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$"),
    "height":     re.compile(r"^(?:1\.\d{2}|2\.00)$"),
    "inn":        re.compile(r"^\d{12}$"),
    "identifier": re.compile(r"^\d{2}-\d{2}/\d{2}$"),
    "occupation": re.compile(
        r"^[А-Яа-яЁёA-Za-z](?:[А-Яа-яЁёA-Za-z \-]*[А-Яа-яЁёA-Za-z])?$"
    ),
    "latitude": re.compile(r"^-?(?:90\.0{3,7}|[0-8]?\d\.\d{3,7})$"),
    "blood_type": re.compile(r"^(?:A|B|AB|O)[+\u2212]$"),
    "issn":       re.compile(r"^\d{4}-\d{4}$"),
    "uuid":       re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
        r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),
    "date":       re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"),
}

VARIANT = 16
CSV_FILE = Path("16.csv")


def is_row_valid(row: dict) -> bool:
    """True, если все поля строки соответствуют своим regex."""
    for column, pattern in PATTERNS.items():
        if pattern.fullmatch(row[column]) is None:
            return False
    return True


def main() -> None:
    invalid_rows: list[int] = []

    
    with CSV_FILE.open("r", encoding="utf-16", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for index, row in enumerate(reader):
            if not is_row_valid(row):
                invalid_rows.append(index)

    print(f"Невалидных строк: {len(invalid_rows)}")

    checksum = calculate_checksum(invalid_rows)
    print(f"Checksum: {checksum}")

    serialize_result(VARIANT, checksum)


if __name__ == "__main__":
    main()