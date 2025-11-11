
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass
from enum import Enum

# Domain Models
class BookStatus(Enum):
    AVAILABLE = "available"
    BORROWED = "borrowed"

@dataclass
class Book:
    title: str
    author: str
    isbn: str
    quantity: int
    available: int

@dataclass
class User:
    name: str
    user_id: str
    email: str
    registration_date: str

@dataclass
class BorrowRecord:
    user_id: str
    isbn: str
    borrow_date: str
    due_date: str
    status: str
    return_date: Optional[str] = None

# Repository Interfaces
class BookRepository(Protocol):
    def find_by_isbn(self, isbn: str) -> Optional[Book]: ...
    def find_by_search_term(self, search_term: str) -> List[Book]: ...
    def save(self, book: Book) -> None: ...
    def find_all(self) -> List[Book]: ...

class UserRepository(Protocol):
    def find_by_id(self, user_id: str) -> Optional[User]: ...
    def save(self, user: User) -> None: ...
    def find_all(self) -> List[User]: ...

class BorrowRepository(Protocol):
    def find_active_by_user_and_isbn(self, user_id: str, isbn: str) -> Optional[BorrowRecord]: ...
    def find_active_by_user(self, user_id: str) -> List[BorrowRecord]: ...
    def find_overdue(self) -> List[BorrowRecord]: ...
    def save(self, record: BorrowRecord) -> None: ...
    def find_all(self) -> List[BorrowRecord]: ...

# Concrete Repositories
class JSONBookRepository:
    def __init__(self, filename: str = 'books.json'):
        self.filename = filename
        self._books: List[Book] = self._load_data()
    
    def _load_data(self) -> List[Book]:
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return [Book(**item) for item in data]
        except:
            return []
    
    def _save_data(self) -> None:
        with open(self.filename, 'w') as f:
            json.dump([book.__dict__ for book in self._books], f)
    
    def find_by_isbn(self, isbn: str) -> Optional[Book]:
        return next((book for book in self._books if book.isbn == isbn), None)
    
    def find_by_search_term(self, search_term: str) -> List[Book]:
        search_term = search_term.lower()
        return [
            book for book in self._books 
            if (search_term in book.title.lower() or 
                search_term in book.author.lower() or 
                search_term == book.isbn)
        ]
    
    def save(self, book: Book) -> None:
        existing_book = self.find_by_isbn(book.isbn)
        if existing_book:
            self._books.remove(existing_book)
        self._books.append(book)
        self._save_data()
    
    def find_all(self) -> List[Book]:
        return self._books.copy()

class JSONUserRepository:
    def __init__(self, filename: str = 'users.json'):
        self.filename = filename
        self._users: List[User] = self._load_data()
    
    def _load_data(self) -> List[User]:
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return [User(**item) for item in data]
        except:
            return []
    
    def _save_data(self) -> None:
        with open(self.filename, 'w') as f:
            json.dump([user.__dict__ for user in self._users], f)
    
    def find_by_id(self, user_id: str) -> Optional[User]:
        return next((user for user in self._users if user.user_id == user_id), None)
    
    def save(self, user: User) -> None:
        existing_user = self.find_by_id(user.user_id)
        if existing_user:
            self._users.remove(existing_user)
        self._users.append(user)
        self._save_data()
    
    def find_all(self) -> List[User]:
        return self._users.copy()

class JSONBorrowRepository:
    def __init__(self, filename: str = 'borrowed.json'):
        self.filename = filename
        self._records: List[BorrowRecord] = self._load_data()
    
    def _load_data(self) -> List[BorrowRecord]:
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                return [BorrowRecord(**item) for item in data]
        except:
            return []
    
    def _save_data(self) -> None:
        with open(self.filename, 'w') as f:
            json.dump([record.__dict__ for record in self._records], f)
    
    def find_active_by_user_and_isbn(self, user_id: str, isbn: str) -> Optional[BorrowRecord]:
        return next(
            (record for record in self._records 
             if record.user_id == user_id and record.isbn == isbn and record.status == 'borrowed'),
            None
        )
    
    def find_active_by_user(self, user_id: str) -> List[BorrowRecord]:
        return [
            record for record in self._records 
            if record.user_id == user_id and record.status == 'borrowed'
        ]
    
    def find_overdue(self) -> List[BorrowRecord]:
        today = datetime.now()
        return [
            record for record in self._records 
            if (record.status == 'borrowed' and 
                today > datetime.strptime(record.due_date, "%Y-%m-%d"))
        ]
    
    def save(self, record: BorrowRecord) -> None:
        self._records.append(record)
        self._save_data()
    
    def find_all(self) -> List[BorrowRecord]:
        return self._records.copy()

# Services
class BookService:
    def __init__(self, book_repo: BookRepository):
        self.book_repo = book_repo
    
    def add_book(self, title: str, author: str, isbn: str, quantity: int = 1) -> str:
        existing_book = self.book_repo.find_by_isbn(isbn)
        
        if existing_book:
            existing_book.quantity += quantity
            existing_book.available += quantity
            self.book_repo.save(existing_book)
            return f"Updated quantity for {title}. New quantity: {existing_book.quantity}"
        
        new_book = Book(
            title=title,
            author=author,
            isbn=isbn,
            quantity=quantity,
            available=quantity
        )
        self.book_repo.save(new_book)
        return f"Added new book: {title}"
    
    def search_books(self, search_term: str) -> List[Book]:
        return self.book_repo.find_by_search_term(search_term)

class UserService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
    
    def register_user(self, name: str, user_id: str, email: str) -> str:
        if self.user_repo.find_by_id(user_id):
            return "User ID already exists"
        
        new_user = User(
            name=name,
            user_id=user_id,
            email=email,
            registration_date=datetime.now().strftime("%Y-%m-%d")
        )
        self.user_repo.save(new_user)
        return f"Registered user: {name}"

class BorrowService:
    def __init__(self, 
                 borrow_repo: BorrowRepository,
                 book_repo: BookRepository,
                 user_repo: UserRepository):
        self.borrow_repo = borrow_repo
        self.book_repo = book_repo
        self.user_repo = user_repo
    
    def borrow_book(self, user_id: str, isbn: str, days: int = 14) -> str:
        user = self.user_repo.find_by_id(user_id)
        if not user:
            return "User not found"
        
        book = self.book_repo.find_by_isbn(isbn)
        if not book:
            return "Book not found"
        
        if book.available <= 0:
            return "Book not available"
        
        existing_borrowing = self.borrow_repo.find_active_by_user_and_isbn(user_id, isbn)
        if existing_borrowing:
            return "Book already borrowed by this user"
        
        borrow_date = datetime.now()
        due_date = borrow_date + timedelta(days=days)
        
        record = BorrowRecord(
            user_id=user_id,
            isbn=isbn,
            borrow_date=borrow_date.strftime("%Y-%m-%d"),
            due_date=due_date.strftime("%Y-%m-%d"),
            status='borrowed'
        )
        
        book.available -= 1
        self.book_repo.save(book)
        self.borrow_repo.save(record)
        
        return f"Book borrowed successfully. Due date: {due_date.strftime('%Y-%m-%d')}"
    
    def return_book(self, user_id: str, isbn: str) -> str:
        record = self.borrow_repo.find_active_by_user_and_isbn(user_id, isbn)
        if not record:
            return "No active borrowing record found"
        
        book = self.book_repo.find_by_isbn(isbn)
        if not book:
            return "Book not found in catalog"
        
        record.status = 'returned'
        record.return_date = datetime.now().strftime("%Y-%m-%d")
        book.available += 1
        
        self.book_repo.save(book)
        # Note: In a real implementation, we'd update the record in the repository
        
        due_date = datetime.strptime(record.due_date, "%Y-%m-%d")
        return_date = datetime.now()
        
        if return_date > due_date:
            days_late = (return_date - due_date).days
            fine = days_late * 5
            return f"Book returned {days_late} days late. Fine: {fine} units"
        
        return "Book returned successfully"
    
    def get_user_borrowings(self, user_id: str) -> List[Dict]:
        borrowings = self.borrow_repo.find_active_by_user(user_id)
        result = []
        
        for record in borrowings:
            book = self.book_repo.find_by_isbn(record.isbn)
            if book:
                result.append({
                    'title': book.title,
                    'author': book.author,
                    'due_date': record.due_date,
                    'status': 'Overdue' if datetime.now() > datetime.strptime(record.due_date, "%Y-%m-%d") else 'On time'
                })
        
        return result
    
    def get_overdue_books(self) -> List[Dict]:
        overdue_records = self.borrow_repo.find_overdue()
        result = []
        
        for record in overdue_records:
            book = self.book_repo.find_by_isbn(record.isbn)
            user = self.user_repo.find_by_id(record.user_id)
            
            if book and user:
                due_date = datetime.strptime(record.due_date, "%Y-%m-%d")
                days_overdue = (datetime.now() - due_date).days
                
                result.append({
                    'title': book.title,
                    'author': book.author,
                    'user_name': user.name,
                    'user_id': user.user_id,
                    'due_date': record.due_date,
                    'days_overdue': days_overdue
                })
        
        return result

# Report Generator
class ReportGenerator:
    def __init__(self, 
                 book_repo: BookRepository,
                 user_repo: UserRepository,
                 borrow_repo: BorrowRepository):
        self.book_repo = book_repo
        self.user_repo = user_repo
        self.borrow_repo = borrow_repo
    
    def generate_library_report(self) -> Dict[str, int]:
        books = self.book_repo.find_all()
        borrowed_records = self.borrow_repo.find_all()
        
        total_books = len(books)
        total_users = len(self.user_repo.find_all())
        currently_borrowed = sum(1 for record in borrowed_records if record.status == 'borrowed')
        overdue_books = len(self.borrow_repo.find_overdue())
        available_books = sum(book.available for book in books)
        
        return {
            'total_books': total_books,
            'total_users': total_users,
            'currently_borrowed': currently_borrowed,
            'overdue_books': overdue_books,
            'available_books': available_books
        }

# Main Application
class LibraryApplication:
    def __init__(self):
        self.book_repo = JSONBookRepository()
        self.user_repo = JSONUserRepository()
        self.borrow_repo = JSONBorrowRepository()
        
        self.book_service = BookService(self.book_repo)
        self.user_service = UserService(self.user_repo)
        self.borrow_service = BorrowService(
            self.borrow_repo, self.book_repo, self.user_repo
        )
        self.report_generator = ReportGenerator(
            self.book_repo, self.user_repo, self.borrow_repo
        )
    
    def run(self):
        while True:
            self._display_menu()
            choice = input("Enter your choice (1-9): ")
            
            if choice == '1':
                self._handle_add_book()
            elif choice == '2':
                self._handle_register_user()
            elif choice == '3':
                self._handle_borrow_book()
            elif choice == '4':
                self._handle_return_book()
            elif choice == '5':
                self._handle_search_books()
            elif choice == '6':
                self._handle_user_borrowings()
            elif choice == '7':
                self._handle_overdue_books()
            elif choice == '8':
                self._handle_generate_report()
            elif choice == '9':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _display_menu(self):
        print("\n=== Library Management System ===")
        print("1. Add Book")
        print("2. Register User")
        print("3. Borrow Book")
        print("4. Return Book")
        print("5. Search Books")
        print("6. User Borrowings")
        print("7. Overdue Books")
        print("8. Generate Report")
        print("9. Exit")
    
    def _handle_add_book(self):
        title = input("Enter book title: ")
        author = input("Enter author: ")
        isbn = input("Enter ISBN: ")
        quantity = int(input("Enter quantity: "))
        result = self.book_service.add_book(title, author, isbn, quantity)
        print(result)
    
    def _handle_register_user(self):
        name = input("Enter user name: ")
        user_id = input("Enter user ID: ")
        email = input("Enter email: ")
        result = self.user_service.register_user(name, user_id, email)
        print(result)
    
    def _handle_borrow_book(self):
        user_id = input("Enter user ID: ")
        isbn = input("Enter ISBN: ")
        days = input("Enter borrow days (default 14): ")
        days = int(days) if days else 14
        result = self.borrow_service.borrow_book(user_id, isbn, days)
        print(result)
    
    def _handle_return_book(self):
        user_id = input("Enter user ID: ")
        isbn = input("Enter ISBN: ")
        result = self.borrow_service.return_book(user_id, isbn)
        print(result)
    
    def _handle_search_books(self):
        search_term = input("Enter search term (title, author, or ISBN): ")
        results = self.book_service.search_books(search_term)
        if results:
            for book in results:
                print(f"Title: {book.title}, Author: {book.author}, ISBN: {book.isbn}, Available: {book.available}")
        else:
            print("No books found")
    
    def _handle_user_borrowings(self):
        user_id = input("Enter user ID: ")
        borrowings = self.borrow_service.get_user_borrowings(user_id)
        if borrowings:
            for borrowing in borrowings:
                print(f"Title: {borrowing['title']}, Due: {borrowing['due_date']}, Status: {borrowing['status']}")
        else:
            print("No borrowings found")
    
    def _handle_overdue_books(self):
        overdue = self.borrow_service.get_overdue_books()
        if overdue:
            for book in overdue:
                print(f"Title: {book['title']}, User: {book['user_name']}, Days overdue: {book['days_overdue']}")
        else:
            print("No overdue books")
    
    def _handle_generate_report(self):
        report = self.report_generator.generate_library_report()
        print(f"Total Books: {report['total_books']}")
        print(f"Total Users: {report['total_users']}")
        print(f"Currently Borrowed: {report['currently_borrowed']}")
        print(f"Overdue Books: {report['overdue_books']}")
        print(f"Available Books: {report['available_books']}")

if __name__ == "__main__":
    app = LibraryApplication()
    app.run()
