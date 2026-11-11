"""Запуск обработки сохранённого каталога книг."""

import sys
from pathlib import Path

from parser import parse_book, parse_books
from storage import save_books


def main() -> int:
    """Читает HTML, сохраняет CSV и возвращает код завершения."""
    source_path = Path(__file__).with_name("catalogue.html")
    output_path = Path(__file__).with_name("books.csv")

    try:
        page_html = source_path.read_text(encoding="utf-8-sig")
        books = parse_books(page_html)
        save_books(books, output_path)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    print(f"Подготовлено записей: {len(books)}")
    print(f"CSV сохранён: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())