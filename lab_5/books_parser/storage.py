"""Сохранение данных о книгах в CSV."""

import csv
from pathlib import Path


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

    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)