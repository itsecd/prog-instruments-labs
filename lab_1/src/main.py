from src.Book import Book
from src.InventoryManager import InventoryManager
from src.ShoppingCart import ShoppingCart
from src.OrderProcessor import OrderProcessor, print_order_details


def display_menu():
    print("Bookstore Management System")
    print("1. View Inventory Manager menu")
    print("2. View Shopping Cart menu")
    print("3. View Order Process menu")
    print("4. Exit")


def display_inventory_menu():
    print("Bookstore Management System / Inventory Manager")
    print("1. View inventory")
    print("2. Add Book")
    print("3. Assign category to a book")
    print("4. Add stock of Books")
    print("5. Remove Book")
    print("6. Search Book")
    print("7. Get Books by category")
    print("8. Update Book details")
    print("9. Generate report")
    print("10. Back to home menu")


def display_cart_menu():
    print("Bookstore Management System / Shopping Cart")
    print("1. View cart")
    print("2. Add Book")
    print("3. Remove book")
    print("4. Empty cart")
    print("5. Update Book quantity")
    print("6. Search Book")
    print("7. Back to home menu")


def display_order_menu():
    print("Bookstore Management System / Order Processor")
    print("1. View orders")
    print("2. Place Order")
    print("3. View last Order")
    print("4. Generate Report")
    print("5. Back to home menu")


class Main:
    def __init__(self):
        self.inventory_manager = InventoryManager()
        self.shopping_cart = ShoppingCart(self.inventory_manager)
        self.order_processor = OrderProcessor(self.shopping_cart, self.inventory_manager)

    def run(self):
        running = True
        while running:
            display_menu()
            choice = input("Enter your choice: ")

            if choice == "1":
                while True:
                    display_inventory_menu()
                    choice2 = input("Enter your choice: ")

                    if choice2 == "1":
                        self.inventory_manager.view_inventory()
                    elif choice2 == "2":
                        self.add_book_to_inventory()
                    elif choice2 == "3":
                        title = input("Enter the title of the book: ")
                        category = input("Enter the new category: ")
                        self.inventory_manager.assign_category(title, category)
                    elif choice2 == "4":
                        title = input("Enter the title of the book: ")
                        quantity = int(input("Enter the quantity of the book: "))
                        self.inventory_manager.add_stock(title, quantity)
                    elif choice2 == "5":
                        self.remove_book_from_inventory()
                    elif choice2 == "6":
                        title = input("Enter the title of the book: ")
                        book = self.inventory_manager.search_book(title)
                        print(book)
                    elif choice2 == "7":
                        category = input("Enter the category of the books: ")
                        books = self.inventory_manager.get_books_by_category(category)

                        if len(books) > 0:
                            for item in books:
                                print(item)
                        else:
                            print("Category not found")
                    elif choice2 == "8":
                        self.update_book_details()
                    elif choice2 == "9":
                        self.inventory_manager.generate_report()
                    elif choice2 == "10":
                        main.run()
                        break

            elif choice == "2":
                while True:
                    display_cart_menu()
                    choice2 = input("Enter your choice: ")

                    if choice2 == "1":
                        self.shopping_cart.view_cart()
                    elif choice2 == "2":
                        self.add_book_to_cart()
                    elif choice2 == "3":
                        self.remove_book_from_cart()
                    elif choice2 == "4":
                        self.shopping_cart.empty_cart()
                    elif choice2 == "5":
                        title = input("Enter the title of the book: ")
                        quantity = int(input("Enter the quantity of the book: "))
                        self.shopping_cart.update_quantity(title, quantity)
                    elif choice2 == "6":
                        title = input("Enter the title of the book: ")
                        book = self.shopping_cart.search_book(title)
                        print(book)
                    elif choice2 == "7":
                        main.run()
                        break

            elif choice == "3":
                while True:
                    display_order_menu()
                    choice2 = input("Enter your choice: ")

                    if choice2 == "1":
                        self.order_processor.view_orders()
                    elif choice2 == "2":
                        self.order_processor.place_order()
                        print("Orders on shopping cart processed")
                    elif choice2 == "3":
                        print_order_details(self.order_processor.orders[-1])
                    elif choice2 == "4":
                        self.order_processor.generate_report()
                    elif choice2 == "5":
                        main.run()
                        break

            elif choice == "4":
                print("Thank you for using the Bookstore Management System. Goodbye!")
                running = False
                break
            else:
                print("Invalid choice. Please try again.")

    def add_book_to_inventory(self):
        title = input("Enter the title of the book: ")
        author = input("Enter the author of the book: ")
        price = float(input("Enter the price of the book: "))
        quantity = int(input("Enter the quantity of the book: "))
        category = input("Enter the category of the book (optional, press Enter to skip): ")

        try:
            book = Book(title, author, price, quantity, category)
            self.inventory_manager.add_book(book)
            print(f"{book.title} added to the inventory.")
        except ValueError as e:
            print(str(e))

    def update_book_details(self):
        print("Update Book Details")
        title = input("Enter the title of the book: ")
        details = input("Enter new details as {key: val} Ex. ( { 'price': 10.0 } ): ")

        try:
            # Convert the input string to a dictionary
            new_details = eval(details)
            if not isinstance(new_details, dict):
                raise ValueError("Invalid input format. Please enter details as a dictionary.")

            # Call the method to update book details
            self.inventory_manager.update_book_details(title, new_details)
            print("Book details updated successfully.")
        except Exception as e:
            print("Error:", e)

    def add_book_to_cart(self):
        title = input("Enter the title of the book: ")
        quantity = int(input("Enter the quantity of the book: "))

        try:
            self.shopping_cart.add_book(title, quantity)
            print(f"{title} added to the cart.")
        except ValueError as e:
            print(str(e))

    def remove_book_from_inventory(self):
        book_title = input("Enter the title of the book to remove: ")
        quantity = int(input("Enter the quantity to remove: "))
        try:
            self.inventory_manager.remove_book(book_title, quantity)
            print(f"{quantity} {book_title}(s) removed from the inventory.")
        except ValueError as e:
            print(str(e))

    def remove_book_from_cart(self):
        book_title = input("Enter the title of the book to remove from the cart: ")
        quantity = int(input("Enter the quantity to remove from the cart: "))
        try:
            self.shopping_cart.remove_book(book_title, quantity)
            print(f"{quantity} {book_title}(s) removed from the cart.")
        except ValueError as e:
            print(str(e))


if __name__ == "__main__":
    main = Main()
    main.run()
