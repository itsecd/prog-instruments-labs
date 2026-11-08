"""Validate variant 26 CSV data and save the assignment checksum."""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from checksum import calculate_checksum
from patterns import PATTERNS


BASE_DIR = Path(__file__).resolve().parent
VARIANT = 26


@dataclass
class ValidationResult:
    """Validation statistics and zero-based indices of invalid data rows."""

    row_count: int
    invalid_rows: list[int]
    errors_by_field: dict[str, int]


def _is_valid_date(value: str) -> bool:
    """Check that the date is a real calendar date in YYYY-MM-DD form."""
    if PATTERNS["date"].fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def validate_csv(path: Path, encoding: str = "utf-16") -> ValidationResult:
    """Check every cell without changing its original contents."""
    result = ValidationResult(0, [], dict.fromkeys(PATTERNS, 0))

    with path.open(encoding=encoding, newline="") as source:
        reader = csv.DictReader(source, delimiter=";", strict=True)
        headers = reader.fieldnames
        if (
            headers is None
            or len(headers) != len(PATTERNS)
            or set(headers) != set(PATTERNS)
        ):
            raise ValueError(
                "CSV headers must match the ten variant 26 fields."
            )

        for row_number, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"Data row {row_number} has an incorrect number of cells."
                )

            invalid = False
            for field, pattern in PATTERNS.items():
                value = row[field]
                if field == "date":
                    valid = _is_valid_date(value)
                else:
                    valid = pattern.fullmatch(value) is not None
                if not valid:
                    result.errors_by_field[field] += 1
                    invalid = True

            if invalid:
                result.invalid_rows.append(row_number)
            result.row_count += 1

    return result


def save_result(path: Path, checksum: str) -> None:
    """Preserve the result.json structure supplied by the assignment."""
    result = {"variant": str(VARIANT), "checksum": checksum}
    path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    """Read paths and CSV encoding from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=BASE_DIR / "26.csv",
        help="Input CSV path; defaults to 26.csv beside this script.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-16",
        help="CSV encoding; defaults to utf-16 for the original dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "result.json",
        help="Result JSON path; defaults to result.json beside this script.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the file, calculate the checksum and print statistics."""
    args = parse_arguments()
    result = validate_csv(args.csv_path, args.encoding)
    checksum = calculate_checksum(result.invalid_rows.copy())
    save_result(args.output, checksum)

    print(f"Data rows: {result.row_count}")
    for field, count in result.errors_by_field.items():
        print(f"{field}: {count}")
    print(f"Invalid cells: {sum(result.errors_by_field.values())}")
    print(f"Invalid rows: {len(result.invalid_rows)}")
    print(f"Checksum: {checksum}")
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()