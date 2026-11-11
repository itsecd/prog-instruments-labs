"""Разбор HTML-каталога и извлечение данных о книгах."""

import re
from html import unescape
from urllib.parse import urljoin
import logging
from logging_config import APP_NAME

logger = logging.getLogger(f"{APP_NAME}.parser")
BASE_URL = "https://books.toscrape.com/"
RATINGS = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def _required_match(
    pattern: str, text: str, error_message: str
) -> re.Match[str]:
    """Находит обязательное совпадение или сообщает об ошибке."""
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(error_message)
    return match


def _normalize_text(value: str) -> str:
    """Раскрывает HTML-сущности и нормализует пробелы."""
    return " ".join(unescape(value).split())


def _parse_title_and_url(card_html: str) -> tuple[str, str]:
    """Извлекает название и формирует полную ссылку."""
    match = _required_match(
        r'<h3><a\s+href="([^"]+)"\s+title="([^"]+)">',
        card_html,
        "не найдены название и ссылка",
    )

    relative_url = unescape(match.group(1)).strip()
    title = _normalize_text(match.group(2))
    if not relative_url or not title:
        raise ValueError("пустые название или ссылка")

    return title, urljoin(BASE_URL, relative_url)


def _parse_price(card_html: str) -> tuple[str, str]:
    """Извлекает цену и валюту."""
    match = _required_match(
        r'<p\s+class="price_color">\s*(£)([0-9]+\.[0-9]{2})\s*</p>',
        card_html,
        "не найдены цена и валюта",
    )
    return match.group(2), match.group(1)


def _parse_rating(card_html: str) -> int:
    """Преобразует словесный рейтинг в число."""
    match = _required_match(
        r'<p\s+class="star-rating\s+(One|Two|Three|Four|Five)">',
        card_html,
        "не найден допустимый рейтинг",
    )
    return RATINGS[match.group(1)]


def _parse_availability(card_html: str) -> str:
    """Извлекает и очищает информацию о наличии."""
    match = _required_match(
        r'<p\s+class="instock\s+availability">\s*'
        r'<i\b[^>]*>\s*</i>\s*([^<]+?)\s*</p>',
        card_html,
        "не найдена информация о наличии",
    )

    availability = _normalize_text(match.group(1))
    if not availability:
        raise ValueError("пустая информация о наличии")

    return availability


def parse_book(card_html: str, number: int) -> dict[str, str | int]:
    """Собирает данные книги и добавляет номер карточки к ошибкам."""
    try:
        title, url = _parse_title_and_url(card_html)
        price, currency = _parse_price(card_html)
        rating = _parse_rating(card_html)
        availability = _parse_availability(card_html)
    except ValueError as error:
        raise ValueError(f"В карточке №{number} {error}") from error
    logger.debug(
        "BOOK_PARSED",
        extra={
            "data": {
                "card_number": number,
                "url": url,
                "rating": rating,
            }
        },
    )
    return {
        "title": title,
        "price": price,
        "currency": currency,
        "rating": rating,
        "availability": availability,
        "url": url,
    }


def parse_books(page_html: str) -> list[dict[str, str | int]]:
    """Извлекает все книги; отклоняет HTML без подходящих карточек."""
    cards = re.findall(
        r'<article class="product_pod">(.*?)</article>',
        page_html,
        flags=re.DOTALL,
    )
    logger.info(
        "CARDS_FOUND",
        extra={"data": {"count": len(cards)}},
    )
    if not cards:
        raise ValueError(
            "Карточки книг не найдены. Проверьте содержимое catalogue.html."
        )

    books = []
    for number, card_html in enumerate(cards, start=1):
        books.append(parse_book(card_html, number))
    return books