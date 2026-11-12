import uuid
import datetime
from src.ShoppingCart import ShoppingCart
from src.InventoryManager import InventoryManager
from src.Book import Book


def generate_order_id():
    """
    Generate a unique order ID.

    Returns:
        UUID: A unique order ID.
    """
    return uuid.uuid4()


def print_order_details(order):
    """
    Print the details of an order.

    Args:
        order (dict): The order dictionary containing order details.
    """
    print("Order ID:", order["order_id"])
    print("Order Date:", order["order_date"])
    print("Book Purchased:")
    for item in order["books"]:
        print("Title:", item.title)
        print("Price:", item.price)
        print("Quantity:", item.quantity)
        print("Total Price for the order:", item.get_total_price())
        print()


class OrderProcessor:
    """
    Process orders placed from a shopping cart.
    """

    def __init__(self, shopping_cart, inventory):
        """
        Initialize the OrderProcessor.

        Args:
            shopping_cart (ShoppingCart): The shopping cart from which orders are processed.
            inventory (InventoryManager): The inventory manager for managing book inventory.
        """
        if not isinstance(shopping_cart, ShoppingCart):
            raise TypeError("Invalid type for 'shopping_cart'. Expected a ShoppingCart object.")
        if not isinstance(inventory, InventoryManager):
            raise TypeError("Invalid type for 'inventory'. Expected an Inventory object.")

        self.orders = []
        self.shopping_cart = shopping_cart
        self.inventory = inventory

    def place_order(self):
        """
        Place an order using the shopping cart contents.
        """
        order_id = generate_order_id()
        order_date = datetime.datetime.now()
        books = [book for book in self.shopping_cart.books]
        total_price = self.shopping_cart.total_price

        if not isinstance(books, list):
            raise TypeError("Invalid type for 'books'. Expected a list.")
        if not all(isinstance(book, Book) for book in books):
            raise TypeError("Invalid type for items in 'books'. Expected Book objects.")
        if not isinstance(total_price, (int, float)) or total_price < 0:
            raise ValueError("Invalid total price. Must be a non-negative number.")

        order = {
            "order_id": order_id,
            "order_date": order_date,
            "books": books,
            "total_price": total_price
        }
        self.orders.append(order)
        self.process_order(order)

    def process_order(self, order):
        """
        Process an order by removing books from inventory and shopping cart.

        Args:
            order (dict): The order to process.
        """
        for book in order['books']:
            self.shopping_cart.remove_book(book.title, book.quantity)
            self.inventory.remove_book(book.title, book.quantity)

    def view_orders(self):
        """
        View the details of all orders.
        """
        if len(self.orders) == 0:
            print("No orders added yet")
        for order in self.orders:
            print_order_details(order)

    def generate_report(self):
        """
        Generate and print a report of total orders and sales.
        """
        total_orders = len(self.orders)
        total_sales = sum(order["total_price"] for order in self.orders)
        print("Total Orders:", total_orders)
        print("Total Sales:", total_sales)
