import csv
import re
from checksum import calculate_checksum, serialize_result


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

