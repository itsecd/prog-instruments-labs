"""Read CSV rows with the delimiter and strictness used by the assignment."""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_rows(
    path: Path, encoding: str = "utf-16"
) -> tuple[list[str], list[dict[str, str]]]:
    """Return (headers, rows) parsed from a semicolon-separated CSV file."""
    logger.debug("Reading CSV file: path=%s, encoding=%s", path, encoding)
    with path.open(encoding=encoding, newline="") as source:
        reader = csv.DictReader(source, delimiter=";", strict=True)
        headers = reader.fieldnames or []
        rows = list(reader)
    logger.info("CSV loaded: rows=%d, headers=%d", len(rows), len(headers))
    return list(headers), rows