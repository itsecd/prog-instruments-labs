import json
import os

class Book:

    def __init__(self, book_id, title, author, is_issued=False):
        self.book_id = book_id
        self.title = title
        self.author = author
        self.is_issued = is_issued

    def to_dict(self):
        return {
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "is_issued": self.is_issued
        }

    @staticmethod
    def from_dict(data):

        return Book(
            str(data.get("book_id", "")),
            data.get("title") or "Unknown",
            data.get("author") or "Unknown",
            bool(data.get("is_issued", False))
        )

    def __str__(self):

        status = "Issued" if self.is_issued else "Available"

        return f"ID: {self.book_id} | {self.title} by {self.author} [{status}]"

class Library:

    def __init__(self):

        self.books = {}
        self.filename = "books.json"

        self.load_books()

    # ---------------- SAVE ----------------

    def save_books(self):

        try:

            data = [
                book.to_dict()
                for book in self.books.values()
            ]

            with open(self.filename, "w") as file:
                json.dump(data, file, indent=4)

            return True

        except IOError:

            print("❌ Unable to save data.")
            return False

    # ---------------- LOAD ----------------

    def load_books(self):

        if not os.path.exists(self.filename):
            return

        try:

            with open(self.filename, "r") as file:
                data = json.load(file)

            if not isinstance(data, list):

                print("⚠️ Invalid database format.")
                return

            for item in data:

                if isinstance(item, dict):

                    book = Book.from_dict(item)

                    if book.book_id:
                        self.books[book.book_id] = book

        except json.JSONDecodeError:

            print("⚠️ books.json corrupted. Starting fresh.")

        except IOError:

            print("❌ Cannot read database.")

    # ---------------- ADD BOOK ----------------

    def add_book(self, book_id, title, author):

        book_id = book_id.strip()
        title = title.strip()
        author = author.strip()

        if not book_id.isdigit():

            print("\n❌ Book ID must contain numbers only.")
            return

        # Normalize IDs
        book_id = str(int(book_id))

        if not title or not author:

            print("\n❌ Title and author cannot be empty.")
            return

        if book_id in self.books:

            print("\n❌ Book ID already exists.")
            return

        self.books[book_id] = Book(
            book_id,
            title,
            author
        )

        if self.save_books():

            print(f"\n✅ '{title}' added successfully.")

    # ---------------- VIEW BOOKS ----------------

    def view_books(self):

        if not self.books:

            print("\n📚 No books available.")
            return

        print("\n========== LIBRARY BOOKS ==========")

        for book_id in sorted(self.books):

            print(self.books[book_id])

    # ---------------- ISSUE BOOK / BORROW BOOK ----------------

    def issue_book(self, book_id):

        book_id = book_id.strip()

        if book_id.isdigit():
            book_id = str(int(book_id))

        book = self.books.get(book_id)

        if not book:

            print("\n❌ Book not found.")

        elif book.is_issued:

            print("\n⚠️ Book already issued.")

        else:

            book.is_issued = True

            self.save_books()

            print(f"\n✅ '{book.title}' issued successfully.")

    # ---------------- RETURN BOOK ----------------

    def return_book(self, book_id):

        book_id = book_id.strip()

        if book_id.isdigit():
            book_id = str(int(book_id))

        book = self.books.get(book_id)

        if not book:

            print("\n❌ Book not found.")

        elif not book.is_issued:

            print("\n⚠️ Book was not issued.")

        else:

            book.is_issued = False

            self.save_books()

            print(f"\n✅ '{book.title}' returned successfully.")

    # ---------------- SEARCH ----------------

    def search_book(self, keyword):

        keyword = keyword.lower().strip()

        if not keyword:

            print("\n❌ Search cannot be empty.")
            return

        found = False

        print("\n========== SEARCH RESULTS ==========")

        for book in self.books.values():

            if (
                keyword in book.book_id.lower()
                or keyword in book.title.lower()
                or keyword in book.author.lower()
            ):

                print(book)
                found = True

        if not found:

            print("❌ No matching book found.")

    # ---------------- DELETE ----------------

    def delete_book(self, book_id):

        book_id = book_id.strip()

        if book_id.isdigit():
            book_id = str(int(book_id))

        book = self.books.get(book_id)

        if not book:

            print("\n❌ Book not found.")

        elif book.is_issued:

            print("\n⚠️ Cannot delete issued book.")

        else:

            confirm = input(
                f"Delete '{book.title}'? (y/n): "
            )

            if confirm.lower() == "y":

                del self.books[book_id]

                self.save_books()

                print("\n✅ Book deleted.")

            else:

                print("\n❌ Delete cancelled.")

# ---------------- MAIN ----------------

def main():

    library = Library()

    while True:

        print("""
===================================
      LIBRARY MANAGEMENT SYSTEM
===================================
1. View Books
2. Add Book
3. Issue Book
4. Search Book
5. Return Book
6. Delete Book
7. Exit
""")

        choice = input("Enter choice (1-7): ").strip()

        if choice == "1":

            library.view_books()

        elif choice == "2":

            book_id = input("Book ID: ")
            title = input("Book Title: ")
            author = input("Author: ")

            library.add_book(
                book_id,
                title,
                author
            )

        elif choice == "3":

            library.issue_book(
                input("Book ID to issue: ")
            )

        elif choice == "4":

            library.search_book(
                input("Search ID/title/author: ")
            )

        elif choice == "5":

            library.return_book(
                input("Book ID to return: ")
            )

        elif choice == "6":

            library.delete_book(
                input("Book ID to delete: ")
            )

        elif choice == "7":

            print("\n👋 Thank you for using Library System.")
            break

        else:

            print("\n❌ Invalid choice.")

if __name__ == "__main__":
    main()