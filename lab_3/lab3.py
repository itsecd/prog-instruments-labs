import csv
import json
import re

from checksum import calculate_checksum

PATTERNS = {
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "height": re.compile(r"^[0-2]\.\d{2}$"),
    "snils": re.compile(r"^\d{11}$"),
    "passport": re.compile(r"^\d{2} \d{2} \d{6}$"),
    "occupation": re.compile(r"^[А-Яа-яЁёA-Za-z\s-]+$"),
    "longitude": re.compile(r"^-?(?:180(?:\.0+)?|(?:1[0-7]\d|\d{1,2})(?:\.\d+)?)$"),
    "hex_color": re.compile(r"^#[0-9a-fA-F]{6}$"),
    "issn": re.compile(r"^\d{4}-\d{4}$"),
    "locale_code": re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$"),
    "time": re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$"),
}


def is_valid_height(value: str) -> bool:
    if not PATTERNS["height"].fullmatch(value):
        return False
    return 0.5 <= float(value) <= 2.5


def is_valid_longitude(value: str) -> bool:
    if not PATTERNS["longitude"].fullmatch(value):
        return False
    return -180.0 <= float(value) <= 180.0


def is_row_valid(row: dict) -> bool:
    for field in ["email", "snils", "passport", "occupation", "hex_color",
                  "issn", "locale_code", "time"]:
        if not PATTERNS[field].fullmatch(row[field]):
            return False

    if not is_valid_height(row["height"]):
        return False
    if not is_valid_longitude(row["longitude"]):
        return False

    return True


def main(csv_path: str = "35.csv") -> None:
    invalid_rows = []

    with open(csv_path, "r", encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for idx, row in enumerate(reader):
            if not is_row_valid(row):
                invalid_rows.append(idx)

    checksum = calculate_checksum(invalid_rows)
    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")

    result = {"variant": 35, "checksum": checksum}
    with open("result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()