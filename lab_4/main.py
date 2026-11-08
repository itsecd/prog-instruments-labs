"""Validate variant 26 CSV data and save the assignment checksum."""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from checksum import calculate_checksum
from columns import COLUMNS, COLUMN_NAMES
from reader import read_rows

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VARIANT = 26


@dataclass
class ValidationResult:
    """Validation statistics and zero-based indices of invalid data rows."""

    row_count: int = 0
    invalid_rows: list[int] = field(default_factory=list)
    errors_by_field: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(COLUMN_NAMES, 0)
    )

def validate_row(row: dict[str, str]) -> list[str]:
    """Return the names of invalid fields in a single CSV row."""
    return [column.name for column in COLUMNS if not column.validator(row[column.name])]

def _validate_headers(headers: list[str]) -> None:
    """Raise if CSV headers don't match the expected columns."""
    if set(headers) != set(COLUMN_NAMES):
        raise ValueError("CSV headers must match the ten variant 26 fields.")


def _apply_row(
    result: ValidationResult, row_number: int, row: dict[str, str]
) -> None:
    """Update result counters for a single CSV row."""
    if None in row or any(value is None for value in row.values()):
        raise ValueError(
            f"Data row {row_number} has an incorrect number of cells."
        )
    invalid_fields = validate_row(row)
    for name in invalid_fields:
        result.errors_by_field[name] += 1
    if invalid_fields:
        result.invalid_rows.append(row_number)
    result.row_count += 1


def validate_csv(path: Path, encoding: str = "utf-16") -> ValidationResult:
    """Check every cell without changing its original contents."""
    result = ValidationResult()
    headers, rows = read_rows(path, encoding)
    _validate_headers(headers)
    for row_number, row in enumerate(rows):
        _apply_row(result, row_number, row)
    return result


def save_result(path: Path, variant: int, checksum: str) -> None:
    """Preserve the result.json structure supplied by the assignment."""
    result = {"variant": str(variant), "checksum": checksum}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read paths, encoding and variant from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=BASE_DIR / f"{DEFAULT_VARIANT}.csv",
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
    parser.add_argument(
        "--variant",
        type=int,
        default=DEFAULT_VARIANT,
        help="Assignment variant number (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the file, calculate the checksum and print statistics."""
    args = parse_arguments(argv)
    result = validate_csv(args.csv_path, args.encoding)
    checksum = calculate_checksum(result.invalid_rows.copy())
    save_result(args.output, args.variant, checksum)

    print(f"Data rows: {result.row_count}")
    for field_name, count in result.errors_by_field.items():
        print(f"{field_name}: {count}")
    print(f"Invalid cells: {sum(result.errors_by_field.values())}")
    print(f"Invalid rows: {len(result.invalid_rows)}")
    print(f"Checksum: {checksum}")
    print(f"Result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())