import argparse
import csv
import re
from pathlib import Path

from checksum import calculate_checksum, serialize_result

VARIANT = 73

PATTERNS: dict[str, str] = {
    "email": r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "http_status_message": r"^[1-5]\d{2} [A-Za-z][A-Za-z '\-]+$",
    "snils": r"^\d{11}$",
    "passport": r"^\d{2} \d{2} \d{6}$",
    "ip_v4": (
        r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
        r"(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
    ),
    "longitude": r"^-?(180(\.0+)?|(1[0-7]\d|[1-9]?\d)(\.\d+)?)$",
    "hex_color": r"^#[0-9a-fA-F]{6}$",
    "isbn": r"^(\d{3}-)?\d-\d{5}-\d{3}-[\dX]$",
    "locale_code": r"^[a-z]{2,3}(-[a-z]{2,4})?$",
    "time": r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$",
}


def parse_arguments() -> argparse.Namespace:
    """Adds and parses command-line arguments."""
    default_file = Path(__file__).parent / f"{VARIANT}.csv"
    parser = argparse.ArgumentParser(description="Validate CSV file using regex")
    parser.add_argument(
        "--file",
        "-f",
        default=str(default_file),
        help="Path to CSV file",
    )
    return parser.parse_args()


def validate_row(row: dict[str, str], patterns: dict[str, str]) -> bool:
    """Checks if all fields in a row match their regex patterns."""
    for column, pattern in patterns.items():
        if not re.fullmatch(pattern, row[column]):
            return False
    return True


def get_invalid_rows(file_path: str) -> list[int]:
    """Reads CSV and returns 0-based indices of invalid data rows."""
    invalid_rows: list[int] = []
    with open(file_path, encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for idx, row in enumerate(reader):
            if not validate_row(row, PATTERNS):
                invalid_rows.append(idx)
    return invalid_rows


def main() -> None:
    args = parse_arguments()
    invalid_rows = get_invalid_rows(args.file)
    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)

    print(f"Invalid rows: {len(invalid_rows)}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
