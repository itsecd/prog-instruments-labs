"""Разбор сохранённой страницы выдачи Avito регулярными выражениями."""

import csv
import html
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "avito_sample.html"
OUTPUT_FILE = BASE_DIR / "results" / "listings.csv"

ITEM_START_RE = re.compile(r'(?=<div[^>]*data-marker="item"[\s>])')
ITEM_ID_RE = re.compile(r'data-item-id="(\d+)"')
TITLE_RE = re.compile(r'data-marker="item-title"[^>]*>([^<]+)</a>')
URL_RE = re.compile(r'data-marker="item-title"[^>]*href="([^"?]+)')
PRICE_RE = re.compile(r'itemprop="price"[^>]*content="(\d+)"')
CITY_RE = re.compile(r'data-marker="item-location".*?<span[^>]*>([^<]+)', re.DOTALL)
# Заголовок вида «ВАЗ (LADA) Vesta 1.6 MT, 2017, 76 000 км»; пробега может не быть.
TITLE_PARTS_RE = re.compile(r"^(.+?),\s*((?:19|20)\d{2})(?:,\s*(\d[\d\s]*?)\s*км)?")

COLUMNS = ["id", "model", "year", "mileage_km", "price_rub", "city", "url"]


def find(pattern: re.Pattern[str], text: str) -> str:
    """Возвращает первую группу первого совпадения или пустую строку."""
    match = pattern.search(text)
    return html.unescape(match[1]).strip() if match else ""


def parse_item(block: str) -> dict[str, str]:
    """Извлекает поля одного объявления из его HTML."""
    title = TITLE_PARTS_RE.match(find(TITLE_RE, block))
    model, year, mileage = title.groups(default="") if title else ("", "", "")
    return {
        "id": find(ITEM_ID_RE, block),
        "model": model,
        "year": year,
        "mileage_km": re.sub(r"\s", "", mileage),
        "price_rub": find(PRICE_RE, block),
        "city": find(CITY_RE, block),
        "url": find(URL_RE, block),
    }


def main() -> None:
    """Читает HTML (путь можно передать аргументом) и сохраняет объявления в CSV."""
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_FILE
    page = input_file.read_text(encoding="utf-8")
    items = [parse_item(block) for block in ITEM_START_RE.split(page)[1:]]

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(items)
    print(f"Saved {len(items)} listings to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
