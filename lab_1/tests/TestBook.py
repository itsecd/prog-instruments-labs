import unittest
from src.Book import Book


class TestBook(unittest.TestCase):

    def test_valid_input(self):
        book = Book("Title", "Author", 10.0, 5, "Category")
        self.assertEqual(book.title, "Title")
        self.assertEqual(book.author, "Author")
        self.assertEqual(book.price, 10.0)
        self.assertEqual(book.quantity, 5)
        self.assertEqual(book.category, "Category")
        self.assertEqual(book.get_total_price(), 10.0 * 5)

    def test_invalid_title(self):
        with self.assertRaises(TypeError):
            Book(123, "Author", 10.0, 5)

    def test_invalid_author(self):
        with self.assertRaises(TypeError):
            Book("Title", 123, 10.0, 5)

    def test_invalid_price(self):
        with self.assertRaises(ValueError):
            Book("Title", "Author", -10.0, 5)

    def test_invalid_quantity(self):
        with self.assertRaises(ValueError):
            Book("Title", "Author", 10.0, -5)

    def test_setters_with_valid_inputs(self):
        book = Book("Title", "Author", 10.0, 5)
        book.title = "New Title"
        self.assertEqual(book.title, "New Title")
        book.author = "New Author"
        self.assertEqual(book.author, "New Author")
        book.price = 20.0
        self.assertEqual(book.price, 20.0)
        book.quantity = 10
        self.assertEqual(book.quantity, 10)
        book.category = "New Category"
        self.assertEqual(book.category, "New Category")

    def test_setters_with_invalid_inputs(self):
        book = Book("Title", "Author", 10.0, 5)
        with self.assertRaises(TypeError):
            book.title = 123
        with self.assertRaises(TypeError):
            book.author = 123
        with self.assertRaises(ValueError):
            book.price = -20.0
        with self.assertRaises(ValueError):
            book.price = "price"
        with self.assertRaises(ValueError):
            book.quantity = -10
        with self.assertRaises(ValueError):
            book.quantity = "quantity"
        with self.assertRaises(TypeError):
            book.category = 123


if __name__ == '__main__':
    unittest.main()
