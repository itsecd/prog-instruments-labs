"""Извлечение данных о книгах из сохранённой страницы Books to Scrape."""

import csv
import re
import sys
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"
RATINGS = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def parse_book(card_html: str, number: int) -> dict[str, str | int]:
    """Разбирает одну карточку или сообщает номер карточки с ошибкой."""
    title_match = re.search(
        r'<h3><a\s+href="([^"]+)"\s+title="([^"]+)">',
        card_html,
    )
    if title_match is None:
        raise ValueError(
            f"В карточке №{number} не найдены название и ссылка"
        )

    relative_url = unescape(title_match.group(1)).strip()
    title = " ".join(unescape(title_match.group(2)).split())
    if not relative_url or not title:
        raise ValueError(
            f"В карточке №{number} пустые название или ссылка"
        )

    price_match = re.search(
        r'<p\s+class="price_color">\s*(£)([0-9]+\.[0-9]{2})\s*</p>',
        card_html,
    )
    if price_match is None:
        raise ValueError(
            f"В карточке №{number} не найдены цена и валюта"
        )

    currency = price_match.group(1)
    price = price_match.group(2)

    rating_match = re.search(
        r'<p\s+class="star-rating\s+(One|Two|Three|Four|Five)">',
        card_html,
    )
    if rating_match is None:
        raise ValueError(
            f"В карточке №{number} не найден допустимый рейтинг"
        )

    rating = RATINGS[rating_match.group(1)]

    availability_match = re.search(
        r'<p\s+class="instock\s+availability">\s*'
        r'<i\b[^>]*>\s*</i>\s*([^<]+?)\s*</p>',
        card_html,
    )
    if availability_match is None:
        raise ValueError(
            f"В карточке №{number} не найдена информация о наличии"
        )

    availability = " ".join(
        unescape(availability_match.group(1)).split()
    )
    if not availability:
        raise ValueError(
            f"В карточке №{number} пустая информация о наличии"
        )

    return {
        "title": title,
        "price": price,
        "currency": currency,
        "rating": rating,
        "availability": availability,
        "url": urljoin(BASE_URL, relative_url),
    }


def parse_books(page_html: str) -> list[dict[str, str | int]]:
    """Извлекает все книги; отклоняет HTML без подходящих карточек."""
    cards = re.findall(
        r'<article class="product_pod">(.*?)</article>',
        page_html,
        flags=re.DOTALL,
    )
    if not cards:
        raise ValueError(
            "Карточки книг не найдены. Проверьте содержимое catalogue.html."
        )

    books = []
    for number, card_html in enumerate(cards, start=1):
        books.append(parse_book(card_html, number))
    return books


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
