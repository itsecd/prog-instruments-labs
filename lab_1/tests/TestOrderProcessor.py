import io
import sys
import unittest
from src.OrderProcessor import OrderProcessor
from src.ShoppingCart import ShoppingCart
from src.InventoryManager import InventoryManager
from src.Book import Book
import uuid
import datetime


class TestOrderProcessor(unittest.TestCase):
    def setUp(self):
        self.inventory = InventoryManager()
        self.shopping_cart = ShoppingCart(self.inventory)

        self.processor = OrderProcessor(self.shopping_cart, self.inventory)
        self.book1 = Book("Book1", "Author1", 10.0, 5)
        self.book2 = Book("Book2", "Author2", 20.0, 10)

        self.inventory.add_book(self.book1)
        self.inventory.add_book(self.book2)
        self.shopping_cart.add_book(self.book1.title, self.book1.quantity)
        self.shopping_cart.add_book(self.book2.title, self.book2.quantity)

    def test_init(self):
        self.assertIsInstance(self.processor.shopping_cart, ShoppingCart)
        self.assertIsInstance(self.processor.inventory, InventoryManager)

    def test_place_order(self):
        self.processor.place_order()
        self.assertEqual(len(self.processor.orders), 1)

        order_books = sorted([book.title for book in self.processor.orders[0]["books"]])
        expected_books = sorted(["Book1", "Book2"])
        self.assertEqual(order_books, expected_books)
        self.assertEqual(self.processor.orders[0]["total_price"], 250.0)

    def test_process_order(self):
        order = {
            "order_id": uuid.uuid4(),
            "order_date": datetime.datetime.now(),
            "books": [self.book1, self.book2],
            "total_price": 250.0
        }
        self.processor.process_order(order)
        self.assertNotIn(self.book1, self.shopping_cart.books)
        self.assertNotIn(self.book2, self.shopping_cart.books)
        self.assertNotIn(self.book1, self.inventory.inventory)
        self.assertNotIn(self.book2, self.inventory.inventory)

    def test_generate_report(self):
        order1 = {"total_price": 100.0}
        order2 = {"total_price": 200.0}
        self.processor.orders = [order1, order2]

        # Capture the output of generate_report
        with io.StringIO() as captured_output:
            sys.stdout = captured_output  # Redirect stdout to captured_output
            self.processor.generate_report()
            sys.stdout = sys.__stdout__  # Reset stdout

            # Get the generated output
            generated_output = captured_output.getvalue()

        # Check if the output contains the expected lines
        expected_output = "Total Orders: 2\nTotal Sales: 300.0\n"
        self.assertEqual(generated_output, expected_output)


if __name__ == '__main__':
    unittest.main()
