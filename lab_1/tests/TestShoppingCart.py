import unittest
from src.ShoppingCart import ShoppingCart
from src.Book import Book
from src.InventoryManager import InventoryManager


class TestShoppingCart(unittest.TestCase):
    def setUp(self):
        self.inventory_manager = InventoryManager()
        self.book1 = Book("Book1", "Author1", 10.0, 5)
        self.book2 = Book("Book2", "Author2", 20.0, 10)
        self.cart = ShoppingCart(self.inventory_manager)

    def test_init(self):
        with self.assertRaises(TypeError):
            self.cart = ShoppingCart("invalid_inventory")

    def test_books_attribute(self):
        self.assertTrue(hasattr(self.cart, "books"))
        self.assertIsInstance(self.cart.books, list)
        self.assertEqual(len(self.cart.books), 0)

    def test_add_book(self):

        with self.assertRaises(ValueError):
            self.cart.add_book(self.book1.title, self.book1.quantity)

        # Add the books to the inventory
        self.inventory_manager.add_book(self.book1)
        self.inventory_manager.add_book(self.book2)

        # Test adding the first book to the cart
        self.cart.add_book(self.book1.title, self.book1.quantity)
        self.assertEqual(len(self.cart.books), 1)
        self.assertEqual(self.cart.total_price, 50.0)

        # Adding the same book should increase the quantity
        self.cart.add_book(self.book1.title, self.book1.quantity)
        self.assertEqual(len(self.cart.books), 1)
        self.assertEqual(self.cart.books[0].quantity, 10)
        self.assertEqual(self.cart.total_price, 100.0)

        # Adding another book
        self.cart.add_book(self.book2.title, self.book2.quantity)
        self.assertEqual(len(self.cart.books), 2)
        self.assertEqual(self.cart.total_price, 300.0)

        # Adding book with invalid type
        with self.assertRaises(TypeError):
            self.cart.add_book(123, 10)

    def test_remove_book(self):

        with self.assertRaises(ValueError):
            self.cart.remove_book(self.book1.title, self.book1.quantity)

        # Adding books to the cart and inventory
        self.inventory_manager.add_book(self.book1)
        self.inventory_manager.add_book(self.book2)

        self.cart.add_book(self.book1.title, self.book1.quantity)
        self.cart.add_book(self.book2.title, self.book2.quantity)

        # Removing a quantity of Book1
        self.cart.remove_book("Book1", 3)  # Removing 3 copies of Book1
        self.assertEqual(len(self.cart.books), 2)  # Still two books in the cart
        self.assertEqual(self.cart.total_price, 220.0)  # Total price updated

        # Removing the remaining Book1
        self.cart.remove_book("Book1", 2)  # Removing the remaining 2 copies of Book1
        self.assertEqual(len(self.cart.books), 1)  # Only one book left in the cart
        self.assertEqual(self.cart.total_price, 200.0)  # Total price updated

        # Removing non-existent book
        with self.assertRaises(ValueError):
            self.cart.remove_book("NonExistentBook", 1)  # Attempting to remove a book not in the cart

        # Removing book with invalid type
        with self.assertRaises(TypeError):
            self.cart.remove_book(123, 1)  # Attempting to remove a book with an invalid type

    def test_view_cart(self):
        self.inventory_manager.add_book(self.book1)
        self.inventory_manager.add_book(self.book2)

        self.cart.add_book(self.book1.title, self.book1.quantity)
        self.cart.add_book(self.book2.title, self.book2.quantity)
        self.cart.view_cart()  # Just to see if it runs without errors

    def test_empty_cart(self):
        self.inventory_manager.add_book(self.book1)
        self.inventory_manager.add_book(self.book2)

        self.cart.add_book(self.book1.title, self.book1.quantity)
        self.cart.add_book(self.book2.title, self.book2.quantity)
        self.cart.empty_cart()
        self.assertEqual(len(self.cart.books), 0)
        self.assertEqual(self.cart.total_price, 0)

    def test_update_quantity(self):
        self.inventory_manager.add_book(self.book1)

        with self.assertRaises(ValueError):
            self.cart.update_quantity(self.book1.title, 3)

        self.cart.add_book(self.book1.title, self.book1.quantity)
        self.cart.update_quantity(self.book1.title, 3)
        self.assertEqual(self.cart.books[0].quantity, 3)
        self.assertEqual(self.cart.total_price, 30.0)

        # Updating quantity with invalid book type
        with self.assertRaises(TypeError):
            self.cart.update_quantity(123, 5)

        # Updating quantity with negative value
        with self.assertRaises(ValueError):
            self.cart.update_quantity(self.book1.title, -5)

    def test_inventory(self):
        self.assertEqual(self.cart.inventory, self.inventory_manager)

    def test_books_setter(self):
        # Setter automatically adds books to cart
        new_books = [self.book1, self.book2]

        with self.assertRaises(ValueError):
            self.cart.books = new_books

        self.inventory_manager.add_book(self.book1)
        self.inventory_manager.add_book(self.book2)

        self.cart.books = new_books
        self.assertEqual(len(self.cart.books), 2)
        self.assertEqual(self.cart.total_price, 250.0)

        # Setting books with invalid type
        with self.assertRaises(TypeError):
            self.cart.books = "invalid_books"

    def test_search_book(self):

        with self.assertRaises(ValueError):
            self.cart.add_book(self.book1.title, self.book1.quantity)
            self.cart.add_book(self.book2.title, self.book2.quantity)

        self.inventory_manager.add_book(self.book1)
        self.inventory_manager.add_book(self.book2)

        self.cart.add_book(self.book1.title, self.book1.quantity)
        self.cart.add_book(self.book2.title, self.book2.quantity)
        found_book = self.cart.search_book("Book1")
        self.assertIsNotNone(found_book)
        self.assertEqual(found_book.title, "Book1")

    def test_search_book_with_invalid_data(self):
        with self.assertRaises(TypeError):
            self.cart.search_book(123)


if __name__ == '__main__':
    unittest.main()
