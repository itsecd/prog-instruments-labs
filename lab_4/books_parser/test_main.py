"""Проверки парсера и записи CSV без обращения к сети."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import main as parser

DATA_DIR = Path(__file__).parent
CARD_HTML = """
<article class="product_pod">
    <h3><a href="catalogue/book/index.html" title="Book">Book</a></h3>
    <p class="star-rating Three"></p>
    <p class="price_color">£51.77</p>
    <p class="instock availability">
        <i class="icon-ok"></i>
        In stock
    </p>
</article>
"""


class BooksParserTests(unittest.TestCase):
    """Проверяет границы карточек, значения полей и ошибки входа."""

    def test_sample_keeps_each_books_fields_together(self):
        page_html = (DATA_DIR / "sample.html").read_text(
            encoding="utf-8-sig"
        )
        books = parser.parse_books(page_html)

        self.assertEqual(len(books), 2)
        self.assertEqual(books[0]["title"], "A Light in the Attic")
        self.assertEqual(books[1]["title"], "A Light in the Attic")
        self.assertEqual(
            [(book["price"], book["rating"], book["availability"])
             for book in books],
            [("51.77", 3, "In stock"),
             ("8.50", 5, "In stock (5 available)")],
        )
        self.assertEqual(
            books[0]["url"],
            "https://books.toscrape.com/"
            "catalogue/a-light-in-the-attic_1000/index.html",
        )

    def test_saved_catalogue_contains_twenty_distinct_books(self):
        page_html = (DATA_DIR / "catalogue.html").read_text(
            encoding="utf-8-sig"
        )
        books = parser.parse_books(page_html)

        self.assertEqual(len(books), 20)
        self.assertEqual(len({book["url"] for book in books}), 20)
        self.assertEqual(books[0]["title"], "A Light in the Attic")
        self.assertEqual(books[-1]["title"], "It's Only the Himalayas")
        self.assertEqual(books[11]["title"], "Shakespeare's Sonnets")

    def test_entities_and_csv_round_trip(self):
        page_html = CARD_HTML.replace(
            'title="Book"',
            'title="Путь, &quot;звёзды&quot; &amp; Shakespeare&#39;s"',
        ).replace(
            "catalogue/book/index.html",
            "catalogue/book/index.html?x=1&amp;y=2",
        ).replace("In stock", "In&nbsp;\n stock")
        books = parser.parse_books(page_html)
        self.assertEqual(
            books[0]["title"], "Путь, \"звёзды\" & Shakespeare's"
        )
        self.assertEqual(books[0]["availability"], "In stock")
        self.assertEqual(
            books[0]["url"],
            "https://books.toscrape.com/"
            "catalogue/book/index.html?x=1&y=2",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "books.csv"
            parser.save_books(books, csv_path)
            with csv_path.open(
                encoding="utf-8-sig", newline=""
            ) as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertEqual(
                    reader.fieldnames,
                    ["title", "price", "currency",
                     "rating", "availability", "url"],
                )
                rows = list(reader)

        expected_row = {key: str(value) for key, value in books[0].items()}
        self.assertEqual(rows, [expected_row])

    def test_invalid_required_fields_are_reported(self):
        cases = [
            ('title="Book"', 'title=""', "название и ссылка"),
            ('title="Book"', 'title="  "', "пустые название"),
            ('href="catalogue/book/index.html"', 'href=" "', "пустые"),
            ("£51.77", "£51x77", "цена и валюта"),
            ("£51.77", "£51.7", "цена и валюта"),
            ("£51.77", "£51.777", "цена и валюта"),
            ("star-rating Three", "star-rating Six", "рейтинг"),
            ("instock availability", "instock missing", "наличии"),
            ("In stock", " \n ", "наличии"),
        ]
        for old, new, message in cases:
            with self.subTest(replacement=new):
                page_html = CARD_HTML + CARD_HTML.replace(old, new)
                with self.assertRaisesRegex(
                    ValueError, f"№2.*{message}"
                ):
                    parser.parse_books(page_html)

    def test_html_without_cards_is_rejected(self):
        for page_html in ("", "<html>Нет карточек</html>"):
            with self.subTest(page_html=page_html):
                with self.assertRaisesRegex(ValueError, "не найдены"):
                    parser.parse_books(page_html)

    def test_invalid_input_does_not_overwrite_previous_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid_html = CARD_HTML + CARD_HTML.replace("£51.77", "bad")
            (root / "catalogue.html").write_text(
                invalid_html, encoding="utf-8"
            )
            csv_path = root / "books.csv"
            csv_path.write_bytes(b"previous result")
            stderr = io.StringIO()

            with patch.object(parser, "__file__", str(root / "main.py")):
                with redirect_stderr(stderr):
                    exit_code = parser.main()

            self.assertEqual(exit_code, 1)
            self.assertIn("№2", stderr.getvalue())
            self.assertEqual(csv_path.read_bytes(), b"previous result")

    def test_missing_input_returns_error_without_creating_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stderr = io.StringIO()
            with patch.object(parser, "__file__", str(root / "main.py")):
                with redirect_stderr(stderr):
                    exit_code = parser.main()

            self.assertEqual(exit_code, 1)
            self.assertIn("Ошибка:", stderr.getvalue())
            self.assertFalse((root / "books.csv").exists())


if __name__ == "__main__":
    unittest.main()
