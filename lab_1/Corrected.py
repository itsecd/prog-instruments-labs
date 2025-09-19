import datetime
import json
import os
from typing import List, Dict, Optional, Union


class Book:
    def __init__(self, title, author, isbn, year, pages):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.year = year
        self.pages = pages
        self.is_available = True

    def get_info(self):
        return f"{self.title} by {self.author} ({self.year}) - {self.pages} pages"

    def __str__(self):
        status = "Available" if self.is_available else "Borrowed"
        return f"{self.title} - {self.author} [{status}]"


class Member:
    def __init__(self, member_id, name, email, phone):
        self.member_id = member_id
        self.name = name
        self.email = email
        self.phone = phone
        self.borrowed_books = []
        self.fines = 0.0

    def add_fine(self, amount):
        self.fines += amount

    def pay_fine(self, amount):
        if amount <= self.fines:
            self.fines -= amount
            return True
        return False

    def can_borrow(self):
        return len(self.borrowed_books) < 5 and self.fines == 0

    def __str__(self):
        return f"Member {self.member_id}: {self.name} ({len(self.borrowed_books)} books borrowed, ${self.fines} fines)"


class Library:
    def __init__(self, name):
        self.name = name
        self.books = []
        self.members = []
        self.transactions = []
        self.loan_period = 14
        self.daily_fine = 0.50

    def add_book(self, book):
        self.books.append(book)

    def find_book_by_isbn(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                return book
        return None

    def find_book_by_title(self, title):
        results = []
        for book in self.books:
            if title.lower() in book.title.lower():
                results.append(book)
        return results

    def register_member(self, member):
        self.members.append(member)

    def find_member_by_id(self, member_id):
        for member in self.members:
            if member.member_id == member_id:
                return member
        return None

    def borrow_book(self, isbn, member_id):
        book = self.find_book_by_isbn(isbn)
        member = self.find_member_by_id(member_id)

        if not book or not member:
            return False, "Book or member not found"

        if not book.is_available:
            return False, "Book is not available"

        if not member.can_borrow():
            return False, "Member cannot borrow more books or has unpaid fines"

        book.is_available = False
        member.borrowed_books.append({
            'book': book,
            'borrow_date': datetime.date.today(),
            'due_date': datetime.date.today() + datetime.timedelta(days=self.loan_period)
        })

        self.transactions.append({
            'type': 'borrow',
            'isbn': isbn,
            'member_id': member_id,
            'date': datetime.date.today(),
            'due_date': datetime.date.today() + datetime.timedelta(days=self.loan_period)
        })

        return True, f"Book borrowed successfully. Due date: {datetime.date.today() + datetime.timedelta(days=self.loan_period)}"

    def return_book(self, isbn, member_id):
        book = self.find_book_by_isbn(isbn)
        member = self.find_member_by_id(member_id)

        if not book or not member:
            return False, "Book or member not found"

        for borrowed in member.borrowed_books:
            if borrowed['book'].isbn == isbn:
                book.is_available = True
                member.borrowed_books.remove(borrowed)

                # Calculate fines if any
                return_date = datetime.date.today()
                if return_date > borrowed['due_date']:
                    days_late = (return_date - borrowed['due_date']).days
                    fine = days_late * self.daily_fine
                    member.add_fine(fine)

                    self.transactions.append({
                        'type': 'return',
                        'isbn': isbn,
                        'member_id': member_id,
                        'date': return_date,
                        'fine': fine,
                        'days_late': days_late
                    })

                    return True, f"Book returned with ${fine:.2f} fine for {days_late} days late"

                self.transactions.append({
                    'type': 'return',
                    'isbn': isbn,
                    'member_id': member_id,
                    'date': return_date
                })

                return True, "Book returned successfully"

        return False, "Member did not borrow this book"

    def search_books(self, query):
        results = []
        for book in self.books:
            if (query.lower() in book.title.lower() or
                    query.lower() in book.author.lower() or
                    query == book.isbn):
                results.append(book)
        return results

    def get_overdue_books(self):
        overdue = []
        today = datetime.date.today()

        for member in self.members:
            for borrowed in member.borrowed_books:
                if today > borrowed['due_date']:
                    overdue.append({
                        'member': member,
                        'book': borrowed['book'],
                        'due_date': borrowed['due_date'],
                        'days_late': (today - borrowed['due_date']).days
                    })

        return overdue

    def save_to_file(self, filename):
        data = {
            'name': self.name,
            'books': [
                {
                    'title': book.title,
                    'author': book.author,
                    'isbn': book.isbn,
                    'year': book.year,
                    'pages': book.pages,
                    'is_available': book.is_available
                } for book in self.books
            ],
            'members': [
                {
                    'member_id': member.member_id,
                    'name': member.name,
                    'email': member.email,
                    'phone': member.phone,
                    'borrowed_books': [
                        {
                            'isbn': b['book'].isbn,
                            'borrow_date': b['borrow_date'].isoformat(),
                            'due_date': b['due_date'].isoformat()
                        } for b in member.borrowed_books
                    ],
                    'fines': member.fines
                } for member in self.members
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename):
        if not os.path.exists(filename):
            return False

        with open(filename, 'r') as f:
            data = json.load(f)

        self.name = data['name']
        self.books = []
        self.members = []

        for book_data in data['books']:
            book = Book(
                book_data['title'],
                book_data['author'],
                book_data['isbn'],
                book_data['year'],
                book_data['pages']
            )
            book.is_available = book_data['is_available']
            self.books.append(book)

        for member_data in data['members']:
            member = Member(
                member_data['member_id'],
                member_data['name'],
                member_data['email'],
                member_data['phone']
            )
            member.fines = member_data['fines']

            for borrowed_data in member_data['borrowed_books']:
                book = self.find_book_by_isbn(borrowed_data['isbn'])
                if book:
                    member.borrowed_books.append({
                        'book': book,
                        'borrow_date': datetime.date.fromisoformat(borrowed_data['borrow_date']),
                        'due_date': datetime.date.fromisoformat(borrowed_data['due_date'])
                    })
                    book.is_available = False

            self.members.append(member)

        return True


def display_menu():
    print("\n" + "=" * 50)
    print("LIBRARY MANAGEMENT SYSTEM")
    print("=" * 50)
    print("1. Add Book")
    print("2. Register Member")
    print("3. Borrow Book")
    print("4. Return Book")
    print("5. Search Books")
    print("6. View All Books")
    print("7. View All Members")
    print("8. View Overdue Books")
    print("9. Pay Fine")
    print("10. Save Data")
    print("11. Load Data")
    print("12. Exit")
    print("=" * 50)


def main():
    library = Library("City Central Library")

    # Add some sample data
    sample_books = [
        Book("The Great Gatsby", "F. Scott Fitzgerald", "978-0-7432-7356-5", 1925, 180),
        Book("To Kill a Mockingbird", "Harper Lee", "978-0-06-112008-4", 1960, 281),
        Book("1984", "George Orwell", "978-0-452-28423-4", 1949, 328),
        Book("Pride and Prejudice", "Jane Austen", "978-0-14-143951-8", 1813, 432),
        Book("The Catcher in the Rye", "J.D. Salinger", "978-0-316-76948-0", 1951, 234)
    ]

    for book in sample_books:
        library.add_book(book)

    while True:
        display_menu()
        choice = input("Enter your choice (1-12): ")

        if choice == "1":
            title = input("Enter book title: ")
            author = input("Enter author: ")
            isbn = input("Enter ISBN: ")
            year = int(input("Enter publication year: "))
            pages = int(input("Enter number of pages: "))

            book = Book(title, author, isbn, year, pages)
            library.add_book(book)
            print("Book added successfully!")

        elif choice == "2":
            member_id = input("Enter member ID: ")
            name = input("Enter name: ")
            email = input("Enter email: ")
            phone = input("Enter phone: ")

            member = Member(member_id, name, email, phone)
            library.register_member(member)
            print("Member registered successfully!")

        elif choice == "3":
            isbn = input("Enter ISBN of book to borrow: ")
            member_id = input("Enter member ID: ")

            success, message = library.borrow_book(isbn, member_id)
            print(message)

        elif choice == "4":
            isbn = input("Enter ISBN of book to return: ")
            member_id = input("Enter member ID: ")

            success, message = library.return_book(isbn, member_id)
            print(message)

        elif choice == "5":
            query = input("Enter search query (title, author, or ISBN): ")
            results = library.search_books(query)

            if results:
                print(f"\nFound {len(results)} book(s):")
                for i, book in enumerate(results, 1):
                    print(f"{i}. {book}")
            else:
                print("No books found.")

        elif choice == "6":
            print(f"\nAll Books ({len(library.books)}):")
            for i, book in enumerate(library.books, 1):
                print(f"{i}. {book}")

        elif choice == "7":
            print(f"\nAll Members ({len(library.members)}):")
            for i, member in enumerate(library.members, 1):
                print(f"{i}. {member}")

        elif choice == "8":
            overdue = library.get_overdue_books()
            if overdue:
                print(f"\nOverdue Books ({len(overdue)}):")
                for i, item in enumerate(overdue, 1):
                    print(f"{i}. {item['book'].title} - Due: {item['due_date']} "
                          f"({item['days_late']} days late) - {item['member'].name}")
            else:
                print("No overdue books.")

        elif choice == "9":
            member_id = input("Enter member ID: ")
            member = library.find_member_by_id(member_id)

            if member:
                print(f"Current fine: ${member.fines:.2f}")
                if member.fines > 0:
                    amount = float(input("Enter amount to pay: "))
                    if member.pay_fine(amount):
                        print(f"Paid ${amount:.2f}. Remaining fine: ${member.fines:.2f}")
                    else:
                        print("Invalid payment amount.")
                else:
                    print("No fines to pay.")
            else:
                print("Member not found.")

        elif choice == "10":
            filename = input("Enter filename to save: ")
            library.save_to_file(filename)
            print("Data saved successfully!")

        elif choice == "11":
            filename = input("Enter filename to load: ")
            if library.load_from_file(filename):
                print("Data loaded successfully!")
            else:
                print("File not found.")

        elif choice == "12":
            print("Thank you for using Library Management System!")
            break

        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()