"""Read CSV rows with the delimiter and strictness used by the assignment."""

import csv
from pathlib import Path


def read_rows(
    path: Path, encoding: str = "utf-16"
) -> tuple[list[str], list[dict[str, str]]]:
    """Return (headers, rows) parsed from a semicolon-separated CSV file."""
    with path.open(encoding=encoding, newline="") as source:
        reader = csv.DictReader(source, delimiter=";", strict=True)
        headers = reader.fieldnames or []
        rows = list(reader)
    return list(headers), rows
