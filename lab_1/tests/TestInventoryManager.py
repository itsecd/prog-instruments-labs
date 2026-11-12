import unittest
from src.InventoryManager import InventoryManager
from src.Book import Book


class TestInventoryManager(unittest.TestCase):
    def setUp(self):
        self.manager = InventoryManager()
        self.book1 = Book("Book1", "Author1", 10.0, 5)
        self.book2 = Book("Book2", "Author2", 20.0, 10)

    def test_add_book(self):
        self.manager.add_book(self.book1)
        self.assertEqual(len(self.manager.inventory), 1)

        book3 = Book("Book1", "Author1", 10.0, 5)
        self.manager.add_book(book3)
        self.assertEqual(len(self.manager.inventory), 1)
        self.assertEqual(self.manager.inventory[0].quantity, 10)

        self.manager.add_book(self.book2)
        self.assertEqual(len(self.manager.inventory), 2)
        self.assertEqual(self.manager.inventory[1].quantity, 10)

    def test_add_book_with_invalid_data(self):
        with self.assertRaises(TypeError):
            self.manager.add_book("Invalid Book Data")

    def test_assign_category(self):
        self.manager.add_book(self.book1)
        self.manager.assign_category("Book1", "Fiction")
        self.assertEqual(self.manager.inventory[0].category, "Fiction")

    def test_assign_category_with_invalid_data(self):
        self.manager.add_book(self.book1)
        with self.assertRaises(TypeError):
            self.manager.assign_category("Book1", 123)

    def test_add_stock(self):
        self.manager.add_book(self.book1)
        self.manager.add_stock("Book1", 5)
        self.assertEqual(self.manager.inventory[0].quantity, 10)

    def test_add_stock_with_invalid_data(self):
        self.manager.add_book(self.book1)
        with self.assertRaises(ValueError):
            self.manager.add_stock("Book1", -5)

    def test_remove_book(self):
        self.manager.add_book(self.book1)
        self.manager.add_book(self.book2)

        self.manager.remove_book("Book1", 2)

        book1 = self.manager.search_book("Book1")
        self.assertEqual(book1.quantity, 3)

        self.manager.remove_book("Book2", 5)

        book2 = self.manager.search_book("Book2")
        self.assertEqual(book2.quantity, 5)

        # Try to remove a non-existent book
        with self.assertRaises(ValueError):
            self.manager.remove_book("NonExistentBook", 1)

        # Try to remove a negative quantity
        with self.assertRaises(ValueError):
            self.manager.remove_book("Book1", -2)

        # Try to remove zero quantity
        with self.assertRaises(ValueError):
            self.manager.remove_book("Book2", 0)

    def test_remove_book_with_invalid_data(self):
        with self.assertRaises(TypeError):
            self.manager.remove_book(123, 5)

    def test_search_book(self):
        self.manager.add_book(self.book1)
        self.manager.add_book(self.book2)
        found_book = self.manager.search_book("Book1")
        self.assertIsNotNone(found_book)
        self.assertEqual(found_book.title, "Book1")

    def test_search_book_with_invalid_data(self):
        with self.assertRaises(TypeError):
            self.manager.search_book(123)

    def test_get_books_by_category(self):
        self.manager.add_book(self.book1)
        self.manager.add_book(self.book2)
        self.manager.assign_category("Book1", "Fiction")
        self.manager.assign_category("Book2", "Non-fiction")
        category_books = self.manager.get_books_by_category("Fiction")
        self.assertEqual(len(category_books), 1)
        self.assertEqual(category_books[0].title, "Book1")

    def test_get_books_by_category_with_invalid_data(self):
        with self.assertRaises(TypeError):
            self.manager.get_books_by_category(123)

    def test_update_book_details(self):
        self.manager.add_book(self.book1)
        self.manager.update_book_details("Book1", {"price": 15.0, "quantity": 8})
        updated_book = self.manager.search_book("Book1")
        self.assertEqual(updated_book.price, 15.0)
        self.assertEqual(updated_book.quantity, 8)

    def test_update_book_details_with_invalid_data(self):
        self.manager.add_book(self.book1)
        with self.assertRaises(TypeError):
            self.manager.update_book_details("Book1", "Invalid Data")

    def test_generate_report(self):
        self.manager.add_book(self.book1)
        self.manager.add_book(self.book2)
        self.manager.generate_report()  # Test that it doesn't throw an error


if __name__ == '__main__':
    unittest.main()
