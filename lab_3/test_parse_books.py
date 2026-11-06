"""Тесты для parse_books.py."""

from pathlib import Path

import pytest

from parse_books import parse_card, parse_page

CARD = """
<div class="image_container">
  <a href="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"><img src="./page1_files/2cdad67c.jpg" alt="x" class="thumbnail"></a>
</div>
<p class="star-rating Three"><i class="icon-star"></i></p>
<h3><a href="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html" title="Tom &amp; Jerry">Tom &amp; ...</a></h3>
<p class="price_color">£51.77</p>
<p class="instock availability">
    <i class="icon-ok"></i>
        In stock
</p>
"""


def test_parse_card_fields() -> None:
    """Все поля карточки извлекаются корректно."""
    book = parse_card(CARD)
    assert book.title == "Tom & Jerry"
    assert book.url.endswith("a-light-in-the-attic_1000/index.html")
    assert book.rating == 3
    assert book.price == 51.77
    assert book.currency == "£"
    assert book.in_stock is True
    assert book.image_file == "2cdad67c.jpg"


def test_parse_card_out_of_stock() -> None:
    """Отсутствие книги определяется по тексту статуса."""
    card = CARD.replace("In stock", "Out of stock")
    assert parse_card(card).in_stock is False


def test_parse_card_missing_price() -> None:
    """Если поле не найдено, выбрасывается понятная ошибка."""
    card = CARD.replace("£51.77", "бесплатно")
    with pytest.raises(ValueError, match="цена"):
        parse_card(card)


def test_parse_page_counts_cards(tmp_path: Path) -> None:
    """Со страницы извлекается столько книг, сколько в ней карточек."""
    page = tmp_path / "page.html"
    cards = f'<article class="product_pod">{CARD}</article>' * 3
    page.write_text(f"<html>{cards}</html>", encoding="utf-8")
    assert len(list(parse_page(page))) == 3