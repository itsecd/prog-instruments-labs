"""Парсинг каталога книг из сохранённых HTML-страниц в CSV через регулярки."""

import csv
import html
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = BASE_DIR / "output" / "books.csv"

# Одна карточка книги целиком (DOTALL, чтобы «.» захватывал переносы строк).
CARD_RE = re.compile(r'<article class="product_pod">(.*?)</article>', re.DOTALL)
# Ссылка и полное название: в тексте ссылки оно обрезано, в title нет.
LINK_RE = re.compile(r'<h3>\s*<a href="(?P<url>[^"]+)"\s+title="(?P<title>[^"]*)"')
# Рейтинг записан словом в классе: star-rating Three.
RATING_RE = re.compile(r'<p class="star-rating (?P<rating>One|Two|Three|Four|Five)">')
# Валюта и сумма с двумя знаками; валюта не цифра, не пробел и не «<».
PRICE_RE = re.compile(
    r'<p class="price_color">\s*(?P<currency>[^\d\s<]+)(?P<amount>\d+\.\d{2})\s*</p>'
)
# Статус наличия: текст после иконки внутри абзаца availability.
STOCK_RE = re.compile(
    r'<p class="[^"]*availability">\s*<i class="[^"]*"></i>\s*(?P<status>[^<]*?)\s*</p>'
)
# Имя файла картинки без локального пути.
IMAGE_RE = re.compile(r'<img src="(?:[^"]*/)?(?P<file>[^/"]+)"')

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


@dataclass(frozen=True)
class Book:
    """Данные одной книги из каталога."""

    title: str
    url: str
    rating: int
    price: float
    currency: str
    in_stock: bool
    image_file: str


def _require(pattern: re.Pattern[str], text: str, what: str) -> re.Match[str]:
    """Ищет шаблон в тексте и выбрасывает ошибку, если совпадения нет."""
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Не найдено поле: {what}")
    return match


def parse_card(card_html: str) -> Book:
    """Разбирает HTML одной карточки в объект Book."""
    link = _require(LINK_RE, card_html, "ссылка и название")
    rating = _require(RATING_RE, card_html, "рейтинг")
    price = _require(PRICE_RE, card_html, "цена")
    stock = _require(STOCK_RE, card_html, "наличие")
    image = _require(IMAGE_RE, card_html, "картинка")

    return Book(
        title=html.unescape(link["title"]),
        url=link["url"],
        rating=RATING_WORDS[rating["rating"]],
        price=float(price["amount"]),
        currency=price["currency"],
        in_stock=stock["status"].lower() == "in stock",
        image_file=image["file"],
    )


def parse_page(path: Path) -> Iterator[Book]:
    """Отдаёт все книги с одной сохранённой страницы."""
    page_html = path.read_text(encoding="utf-8")
    for card in CARD_RE.findall(page_html):
        yield parse_card(card)


def write_csv(books: list[Book], path: Path) -> None:
    """Сохраняет книги в CSV (utf-8-sig, чтобы Excel верно читал символ £)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(Book)])
        writer.writeheader()
        writer.writerows(asdict(book) for book in books)


def main() -> None:
    """Парсит все страницы из data/ и записывает результат в CSV."""
    books = [
        book for page in sorted(DATA_DIR.glob("*.html")) for book in parse_page(page)
    ]
    write_csv(books, OUTPUT_FILE)
    print(f"Книг найдено: {len(books)}. Результат: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()