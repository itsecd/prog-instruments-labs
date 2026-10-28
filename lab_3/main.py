import re
import csv
import argparse
import os


def create_patterns():
    """
    Создает и компилирует регулярные выражения для парсинга HTML.
    Возвращает словарь с паттернами.
    """
    return {
        "block": re.compile(r'<div class="book-card">(.*?)</div>', re.DOTALL),

        "title": re.compile(r'<h3 class="title">(.*?)</h3>'),

        "author": re.compile(r'<span class="author">(.*?)</span>'),

        "year": re.compile(r'<span class="year">(\d{4})</span>'),

        "price": re.compile(r'<span class="price">([\d\s]+)\s*₽</span>')
    }


def parse_books(html_content, patterns):
    """
    Проходит по HTML-контенту и собирает данные о книгах.
    Возвращает список словарей.
    """
    books = []

    blocks = patterns["block"].findall(html_content)

    for block in blocks:
        title_match = patterns["title"].search(block)
        author_match = patterns["author"].search(block)
        year_match = patterns["year"].search(block)
        price_match = patterns["price"].search(block)

        if title_match and author_match and year_match:
            price = 0
            if price_match:
                raw_price = price_match.group(1).replace(' ', '').replace('\xa0', '')
                price = int(raw_price) if raw_price else 0

            book_data = {
                "title": title_match.group(1).strip(),
                "author": author_match.group(1).strip(),
                "year": int(year_match.group(1)),
                "price": price
            }
            books.append(book_data)

    return books


def save_to_csv(books, output_path):
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


def main():
    """Главная функция, точка входа в программу."""
    parser = argparse.ArgumentParser(description="Парсер каталога книг из HTML")
    parser.add_argument(
        "input_file",
        help="Путь к HTML-файлу (например, books.html)"
    )
    parser.add_argument(
        "-o", "--output",
        default="library_catalog.csv",
        help="Путь для сохранения CSV-файла с результатами"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Ошибка: Файл '{args.input_file}' не найден.")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    print(f"Чтение файла {args.input_file} завершено.")

    patterns = create_patterns()
    books = parse_books(html_content, patterns)

    save_to_csv(books, args.output)


if __name__ == "__main__":
    main()