import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Union


DEFAULT_OUTPUT_FILE = "library_catalog.csv"
PATTERN_BLOCK = re.compile(r'<div class="book-card">(.*?)</div>', re.DOTALL)
PATTERN_TITLE = re.compile(r'<h3 class="title">(.*?)</h3>')
PATTERN_AUTHOR = re.compile(r'<span class="author">(.*?)</span>')
PATTERN_YEAR = re.compile(r'<span class="year">(\d{4})</span>')
PATTERN_PRICE = re.compile(r'<span class="price">([\d\s]+)\s*₽</span>')


def parse_book_block(block: str) -> Union[Dict[str, Union[str, int]], None]:
    """Парсит один блок HTML и возвращает словарь с данными книги."""
    title_match = PATTERN_TITLE.search(block)
    author_match = PATTERN_AUTHOR.search(block)
    year_match = PATTERN_YEAR.search(block)
    price_match = PATTERN_PRICE.search(block)

    if not (title_match and author_match and year_match):
        return None

    price = 0
    if price_match:
        raw_price = price_match.group(1).replace(' ', '').replace('\xa0', '')
        price = int(raw_price) if raw_price else 0

    return {
        "title": title_match.group(1).strip(),
        "author": author_match.group(1).strip(),
        "year": int(year_match.group(1)),
        "price": price
    }


def parse_html_content(html_content: str) -> List[Dict[str, Union[str, int]]]:
    """Извлекает все книги из HTML-контента."""
    books = []
    blocks = PATTERN_BLOCK.findall(html_content)
    for block in blocks:
        book_data = parse_book_block(block)
        if book_data:
            books.append(book_data)
    return books


def read_file(filepath: Path) -> str:
    """Читает файл и возвращает его содержимое."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_to_csv(books: List[Dict[str, Union[str, int]]], output_path: Path) -> None:
    """Сохраняет список книг в CSV-файл."""
    if not books:
        print("Нет данных для сохранения.")
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["title", "author", "year", "price"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)
    print(f"Успешно сохранено {len(books)} книг в файл: {output_path}")


def main() -> None:
    """Точка входа в программу."""
    parser = argparse.ArgumentParser(description="Парсер каталога книг из HTML")
    parser.add_argument("input_file", help="Путь к HTML-файлу")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE, help="Путь для сохранения CSV")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Ошибка: Файл '{input_path}' не найден.")
        return

    try:
        print(f"Чтение файла {input_path} завершено.")
        html_content = read_file(input_path)
        books = parse_html_content(html_content)
        save_to_csv(books, output_path)
    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")


if __name__ == "__main__":
    main()