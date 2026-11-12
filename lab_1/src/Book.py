class Book:
    """
    Represents a book.

    Attributes:
        title (str): The title of the book.
        author (str): The author of the book.
        price (float): The price of the book.
        quantity (int): The quantity of the book.
        category (str, optional): The category of the book. Defaults to None.

    """

    def __init__(self, title, author, price, quantity, category=None):
        """
        Initialize a Book object with specified attributes.

        Args:
            title (str): The title of the book.
            author (str): The author of the book.
            price (float): The price of the book.
            quantity (int): The quantity of the book.
            category (str, optional): The category of the book. Defaults to None.

        Raises:
            TypeError: If title, author, or category is not a string.
            ValueError: If price is not a non-negative number, or if quantity is not a non-negative integer.

        """
        if not isinstance(title, str):
            raise TypeError("Title must be a string")
        if not isinstance(author, str):
            raise TypeError("Author must be a string")
        if not isinstance(price, (int, float)) or price < 0:
            raise ValueError("Price must be a non-negative number")
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError("Quantity must be a non-negative integer")

        self._title = title
        self._author = author
        self._price = price
        self._quantity = quantity
        self._category = category

    def __str__(self):
        """
        Return a string representation of the Book object.

        Returns:
            str: String representation of the Book object.

        """
        return (f"Title: {self._title}, Author: {self._author}, Price: {self._price}, Quantity: {self._quantity}, "
                f"Category: {self._category}")

    @property
    def title(self):
        """
        Get the title of the book.

        Returns:
            str: The title of the book.

        """
        return self._title

    @title.setter
    def title(self, title):
        """
        Set the title of the book.

        Args:
            title (str): The new title of the book.

        Raises:
            TypeError: If the provided title is not a string.

        """
        if not isinstance(title, str):
            raise TypeError("Title must be a string")
        self._title = title

    @property
    def author(self):
        """
        Get the author of the book.

        Returns:
            str: The author of the book.

        """
        return self._author

    @author.setter
    def author(self, author):
        """
        Set the author of the book.

        Args:
            author (str): The new author of the book.

        Raises:
            TypeError: If the provided author is not a string.

        """
        if not isinstance(author, str):
            raise TypeError("Author must be a string")
        self._author = author

    @property
    def price(self):
        """
        Get the price of the book.

        Returns:
            float: The price of the book.

        """
        return self._price

    @price.setter
    def price(self, price):
        """
        Set the price of the book.

        Args:
            price (float): The new price of the book.

        Raises:
            ValueError: If the provided price is not a non-negative number.

        """
        if not isinstance(price, (int, float)) or price < 0:
            raise ValueError("Price must be a non-negative number")
        self._price = price

    @property
    def quantity(self):
        """
        Get the quantity of the book.

        Returns:
            int: The quantity of the book.

        """
        return self._quantity

    @quantity.setter
    def quantity(self, quantity):
        """
        Set the quantity of the book.

        Args:
            quantity (int): The new quantity of the book.

        Raises:
            ValueError: If the provided quantity is not a non-negative integer.

        """
        if not isinstance(quantity, int) or quantity < 0:
            raise ValueError("Quantity must be a non-negative integer")
        self._quantity = quantity

    @property
    def category(self):
        """
        Get the category of the book.

        Returns:
            str: The category of the book.

        """
        return self._category

    @category.setter
    def category(self, category):
        """
        Set the category of the book.

        Args:
            category (str): The new category of the book.

        Raises:
            TypeError: If the provided category is not a string.

        """
        if not isinstance(category, str):
            raise TypeError("Category must be a string")
        self._category = category

    def get_total_price(self):
        """
        Calculate the total price of the book.

        Returns:
            float: The total price of the book (price * quantity).

        """
        return self._price * self._quantity
