from src.Book import Book


class InventoryManager:

    def __init__(self):
        """
        Initialize the InventoryManager with an empty inventory.
        """
        self._inventory = []

    def add_book(self, book: Book):
        """
        Add a book to the inventory.

        Args:
            book (Book): An instance of the Book class representing the book to add.
        """
        if not isinstance(book, Book):
            raise TypeError("book must be an instance of Book")

        if not self.search_book(book.title):
            self._inventory.append(book)

        else:
            book_copy = Book(book.title, book.author, book.price, book.quantity, book.category)

            for item in self._inventory:
                if item.title == book_copy.title:
                    item.quantity += book_copy.quantity
                    break

    def assign_category(self, book_title, category):
        """
        Assign a category to a book.

        Args:
            book_title (str): The title of the book.
            category (str): The category to assign to the book.
        """
        if not isinstance(book_title, str):
            raise TypeError("book_title must be a string")
        if not isinstance(category, str):
            raise TypeError("category must be a string")
        for book in self._inventory:
            if book.title == book_title:
                book.category = category
                break

    def view_inventory(self):
        """
        View the inventory, printing out details of each book.
        """
        if len(self._inventory) == 0:
            print("No books added yet")

        for book in self._inventory:
            print("Title:", book.title)
            print("Author:", book.author)
            print("Price:", book.price)
            print("Quantity:", book.quantity)
            print("Category:", book.category)
            print()

    def add_stock(self, book_title, quantity):
        """
        Add a quantity of books to the inventory.

        Args:
            book_title (str): The title of the book.
            quantity (int): The quantity of books to add.
        """
        if not isinstance(book_title, str):
            raise TypeError("book_title must be a string")
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError("quantity must be a non-negative integer")
        for book in self._inventory:
            if book.title == book_title:
                book.quantity += quantity
                break

    def remove_book(self, book_title, quantity):
        """
        Removes a specific quantity of a book from the shopping cart.

        Args:
            book_title (str): The title of the book to remove from the cart.
            quantity (int): The quantity of the book to remove.
        """
        if not isinstance(book_title, str):
            raise TypeError("Book title must be a string")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer")
        if self.search_book(book_title) is None:
            raise ValueError("Book title must be in the inventory")

        for item in self._inventory:
            if item.title == book_title:
                if item.quantity <= quantity:
                    self._inventory.remove(item)
                else:
                    item.quantity -= quantity
                break

    def search_book(self, book_title):
        """
        Search for a book by title.

        Args:
            book_title (str): The title of the book to search for.

        Returns:
            Book or None: The book object if found, None otherwise.
        """
        if not isinstance(book_title, str):
            raise TypeError("book_title must be a string")
        for book in self._inventory:
            if book.title == book_title:
                return book
        return None

    def get_books_by_category(self, category):
        """
        Get all books in a given category.

        Args:
            category (str): The category to filter books by.

        Returns:
            list: List of book objects in the specified category.
        """
        if not isinstance(category, str):
            raise TypeError("category must be a string")
        category_books = []
        for book in self._inventory:
            if book.category == category:
                category_books.append(book)
        return category_books

    def update_book_details(self, book_title, new_details):
        """
        Update details of a book.

        Args:
            book_title (str): The title of the book to update.
            new_details (dict): A dictionary containing the updated details.
        """
        if not isinstance(book_title, str):
            raise TypeError("book_title must be a string")
        if not isinstance(new_details, dict):
            raise TypeError("new_details must be a dictionary")
        for book in self._inventory:
            if book.title == book_title:
                for key, value in new_details.items():
                    setattr(book, key, value)
                break

    def generate_report(self):
        """
        Generate and print an inventory report.
        """
        total_books = len(self._inventory)
        categories = set(book.category for book in self._inventory)

        print("Inventory Report:")
        print("Total Number of Books:", total_books)
        print("Categories:", categories)

    @property
    def inventory(self):
        return self._inventory
