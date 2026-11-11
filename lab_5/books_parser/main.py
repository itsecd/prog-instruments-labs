"""Запуск обработки сохранённого каталога книг."""

import logging
import sys
from pathlib import Path

from logging_config import APP_NAME, configure_logging
from parser import parse_book, parse_books
from storage import save_books

logger = logging.getLogger(f"{APP_NAME}.main")


def main() -> int:
    """Читает HTML, сохраняет CSV и возвращает код завершения."""
    source_path = Path(__file__).with_name("catalogue.html")
    output_path = Path(__file__).with_name("books.csv")

    logger.info(
        "APP_STARTED",
        extra={
            "data": {
                "source": str(source_path),
                "output": str(output_path),
            }
        },
    )

    try:
        page_html = source_path.read_text(encoding="utf-8-sig")
        logger.info(
            "INPUT_READ",
            extra={
                "data": {
                    "path": str(source_path),
                    "characters": len(page_html),
                }
            },
        )

        books = parse_books(page_html)
        save_books(books, output_path)
    except (OSError, UnicodeError, ValueError) as error:
        logger.exception(
            "APP_FAILED",
            extra={
                "data": {
                    "source": str(source_path),
                    "output": str(output_path),
                    "error_type": type(error).__name__,
                }
            },
        )
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    logger.info(
        "APP_FINISHED",
        extra={"data": {"records": len(books), "exit_code": 0}},
    )
    print(f"Подготовлено записей: {len(books)}")
    print(f"CSV сохранён: {output_path}")
    return 0


if __name__ == "__main__":
    log_path = Path(__file__).parent / "logs" / "app.log"
    try:
        configure_logging(log_path)
    except OSError as error:
        print(f"Ошибка настройки журнала: {error}", file=sys.stderr)
        raise SystemExit(1)

    raise SystemExit(main())