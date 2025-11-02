import csv
from checksum import calculate_checksum, serialize_result
from consts import *

def load_data(file_path: str, delimiter: str = CSV_DELIMITER):
    """Reads CSV and returns a list of strings."""
    with open(file_path, "r", encoding=FILE_ENCODING, newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        return list(reader)


def validate_row(row: list[str], patterns: list[re.Pattern]) -> bool:
    """Checks one string for compliance with all patterns."""
    if len(row) != len(patterns):
        return False
    return all(pattern.fullmatch(value.strip()) for pattern, value in zip(patterns, row))


def find_invalid_rows(data: list[list[str]], patterns: list[re.Pattern]) -> list[int]:
    """Returns the number with invalid data."""
    invalid_lines = []
    for index, row in enumerate(data[1:], start=0):  # пропускаем заголовок
        if not validate_row(row, patterns):
            invalid_lines.append(index)
    return invalid_lines


def main():
    print("=== CSV Validation Script ===")
    data = load_data(DEFAULT_FILE_PATH)
    patterns = get_validation_patterns()

    invalid_rows = find_invalid_rows(data, patterns)
    print(f"Invalid rows: ", len(invalid_rows))

    checksum_value = calculate_checksum(invalid_rows)
    serialize_result(DEFAULT_VARIANT, checksum_value)

    print(f"Checksum: {checksum_value}")
    print(f"Result saved to result.json (variant {DEFAULT_VARIANT})")


if __name__ == "__main__":
    main()
