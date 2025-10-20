import csv
import re
from checksum import calculate_checksum, serialize_result
from consts import *


def load_data(file_path: str):
    """Loads CSV and returns a list of rows (list[list[str]])."""
    with open(file_path, "r", encoding="windows-1251", newline="") as file:
        reader = csv.reader(file)
        return list(reader)


def validate_row(row: list[str], patterns: list[re.Pattern]) -> bool:
    """Checks a row against all regex patterns."""
    if len(row) != len(patterns):
        return False
    return all(pattern.fullmatch(value.strip()) for pattern, value in zip(patterns, row))


def get_validation_patterns() -> list[re.Pattern]:
    """Returns 10 regex patterns for validation."""
    patterns = [
        NAME_PATTERN,
        SURNAME_PATTERN,
        POSTAL_CODE_PATTERN,
        EMAIL_PATTERN,
        PHONE_PATTERN,
        DATE_PATTERN,
        PASSPORT_PATTERN,
        DECIMAL_PATTERN,
        ID_PATTERN,
        URL_PATTERN,
    ]
    return [re.compile(pattern) for pattern in patterns]


def find_invalid_rows(data: list[list[str]], patterns: list[re.Pattern]) -> list[int]:
    """Finds invalid data rows (excluding header)."""
    invalid_lines = []
    for index, row in enumerate(data[1:], start=0):
        if not validate_row(row, patterns):
            invalid_lines.append(index)
    return invalid_lines


def main():
    file_path = "13.csv"
    variant = 13

    data = load_data(file_path)
    patterns = get_validation_patterns()
    invalid_rows = find_invalid_rows(data, patterns)

    checksum_value = calculate_checksum(invalid_rows)
    serialize_result(variant, checksum_value)

    print("Checksum:", checksum_value)
    print("Result saved in result.json")


if __name__ == "__main__":
    main()