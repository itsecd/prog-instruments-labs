"""Разбор HTML-каталога и извлечение данных о книгах."""

import re
from html import unescape
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