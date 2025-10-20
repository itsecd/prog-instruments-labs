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


def get_validation_patterns() -> list[re.Pattern]:
    """Returns 10 regex patterns for validation."""
    return [
        re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"), # Name
        re.compile(r"[A-Z][a-z]+"), # Surname
        re.compile(r"[0-9]{2}-[0-9]{3}"), # Postal code
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), # Email
        re.compile(r"\+?\d{1,3}[- ]?\(?\d{2,3}\)?[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}"), # Phone
        re.compile(r"\d{4}-\d{2}-\d{2}"), # Date (YYYY-MM-DD)
        re.compile(r"[A-Z]{2}\d{6}"), # Passport
        re.compile(r"[0-9]+\.[0-9]{2}"), # Decimal number
        re.compile(r"[A-Za-z0-9]{8,}"), # ID / Password
        re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), # URL
    ]


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