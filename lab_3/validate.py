import csv
import re
from pathlib import Path
from typing import Dict, List

from checksum import calculate_checksum, serialize_result

VARIANT = 15
CSV_PATH = Path("15.csv")

PATTERNS = {
    "email": re.compile(
        r"^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*@"
        r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+$"
    ),
    "height": re.compile(r"^(?:1\.\d{2}|2\.00)$"),
    "inn": re.compile(r"^\d{12}$"),
    "passport": re.compile(r"^\d{2} \d{2} \d{6}$"),
    "occupation": re.compile(
        r"^[A-Za-zА-Яа-яЁё]+(?:[ -][A-Za-zА-Яа-яЁё]+)*$"
    ),
    "latitude": re.compile(
        r"^-?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?)$"
    ),
    "hex_color": re.compile(r"^#[0-9A-Fa-f]{6}$"),
    "issn": re.compile(r"^\d{4}-\d{4}$"),
    "uuid": re.compile(
        r"^[0-9A-Fa-f]{8}-"
        r"[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{12}$"
    ),
    "time": re.compile(
        r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$"
    ),
}


def is_valid_row(row: Dict[str, str]) -> bool:
    for field, value in row.items():
        if value is None or not PATTERNS[field].fullmatch(value.strip()):
            return False
    return True


def find_invalid_rows(csv_path: Path) -> List[int]:
    invalid_rows: List[int] = []

    with csv_path.open(encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row_number, row in enumerate(reader):
            # Пропускаем полностью пустые строки
            if not any(v and v.strip() for v in row.values() if v):
                continue

            if not is_valid_row(row):
                invalid_rows.append(row_number)

    return invalid_rows


def main() -> None:
    invalid_rows = find_invalid_rows(CSV_PATH)
    checksum = calculate_checksum(invalid_rows)

    serialize_result(VARIANT, checksum)

    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()