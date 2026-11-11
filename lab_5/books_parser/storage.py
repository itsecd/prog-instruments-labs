"""Сохранение данных о книгах в CSV."""

import csv
import logging
from pathlib import Path

from logging_config import APP_NAME

logger = logging.getLogger(f"{APP_NAME}.storage")


def save_books(
    books: list[dict[str, str | int]], output_path: Path
) -> None:
    """Записывает книги в CSV с заголовком и кодировкой UTF-8 с BOM."""
    fieldnames = [
        "title",
        "price",
        "currency",
        "rating",
        "availability",
        "url",
    ]

    logger.debug(
        "CSV_WRITE_STARTED",
        extra={
            "data": {
                "path": str(output_path),
                "records": len(books),
            }
        },
    )

    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)

    logger.info(
        "CSV_SAVED",
        extra={
            "data": {
                "path": str(output_path),
                "records": len(books),
            }
        },
    )