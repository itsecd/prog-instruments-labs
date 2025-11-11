
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

class LibraryManager:
    def __init__(self):
        self.books = []
        self.users = []
        self.borrowed_books = []
        self.load_data()
    
    def load_data(self):
        try:
            with open('books.json', 'r') as f:
                self.books = json.load(f)
        except:
            self.books = []
        
        try:
            with open('users.json', 'r') as f:
                self.users = json.load(f)
        except:
            self.users = []
        
        try:
            with open('borrowed.json', 'r') as f:
                self.borrowed_books = json.load(f)
        except:
            self.borrowed_books = []
    
    def save_data(self):
        with open('books.json', 'w') as f:
            json.dump(self.books, f)
        with open('users.json', 'w') as f:
            json.dump(self.users, f)
        with open('borrowed.json', 'w') as f:
            json.dump(self.borrowed_books, f)
    
    def add_book(self, title, author, isbn, quantity=1):
        for book in self.books:
            if book['isbn'] == isbn:
                book['quantity'] += quantity
                self.save_data()
                return f"Updated quantity for {title}. New quantity: {book['quantity']}"
        
        new_book = {
            'title': title,
            'author': author,
            'isbn': isbn,
            'quantity': quantity,
            'available': quantity
        }
        self.books.append(new_book)
        self.save_data()
        return f"Added new book: {title}"
    
    def register_user(self, name, user_id, email):
        for user in self.users:
            if user['user_id'] == user_id:
                return "User ID already exists"
        
        new_user = {
            'name': name,
            'user_id': user_id,
            'email': email,
            'registration_date': datetime.now().strftime("%Y-%m-%d")
        }
        self.users.append(new_user)
        self.save_data()
        return f"Registered user: {name}"
    
    def find_book(self, search_term):
        results = []
        for book in self.books:
            if (search_term.lower() in book['title'].lower() or 
                search_term.lower() in book['author'].lower() or 
                search_term == book['isbn']):
                results.append(book)
        return results
    
    def borrow_book(self, user_id, isbn, days=14):
        # Find user
        user = None
        for u in self.users:
            if u['user_id'] == user_id:
                user = u
                break
        
        if not user:
            return "User not found"
        
        # Find book
        book = None
        for b in self.books:
            if b['isbn'] == isbn:
                book = b
                break
        
        if not book:
            return "Book not found"
        
        if book['available'] <= 0:
            return "Book not available"
        
        # Check if already borrowed
        for borrowed in self.borrowed_books:
            if (borrowed['user_id'] == user_id and 
                borrowed['isbn'] == isbn and 
                borrowed['status'] == 'borrowed'):
                return "Book already borrowed by this user"
        
        # Calculate dates
        borrow_date = datetime.now()
        due_date = borrow_date + timedelta(days=days)
        
        borrowed_record = {
            'user_id': user_id,
            'isbn': isbn,
            'borrow_date': borrow_date.strftime("%Y-%m-%d"),
            'due_date': due_date.strftime("%Y-%m-%d"),
            'status': 'borrowed'
        }
        
        self.borrowed_books.append(borrowed_record)
        book['available'] -= 1
        self.save_data()
        
        return f"Book borrowed successfully. Due date: {due_date.strftime('%Y-%m-%d')}"
    
    def return_book(self, user_id, isbn):
        # Find borrowed record
        borrowed_record = None
        for record in self.borrowed_books:
            if (record['user_id'] == user_id and 
                record['isbn'] == isbn and 
                record['status'] == 'borrowed'):
                borrowed_record = record
                break
        
        if not borrowed_record:
            return "No active borrowing record found"
        
        # Find book
        book = None
        for b in self.books:
            if b['isbn'] == isbn:
                book = b
                break
        
        if not book:
            return "Book not found in catalog"
        
        # Update records
        borrowed_record['status'] = 'returned'
        borrowed_record['return_date'] = datetime.now().strftime("%Y-%m-%d")
        book['available'] += 1
        
        # Check for late return
        due_date = datetime.strptime(borrowed_record['due_date'], "%Y-%m-%d")
        return_date = datetime.now()
        
        if return_date > due_date:
            days_late = (return_date - due_date).days
            fine = days_late * 5  # 5 units per day late
            self.save_data()
            return f"Book returned {days_late} days late. Fine: {fine} units"
        
        self.save_data()
        return "Book returned successfully"
    
    def get_user_borrowings(self, user_id):
        user_borrowings = []
        for record in self.borrowed_books:
            if record['user_id'] == user_id:
                # Find book details
                for book in self.books:
                    if book['isbn'] == record['isbn']:
                        borrowing_info = record.copy()
                        borrowing_info['title'] = book['title']
                        borrowing_info['author'] = book['author']
                        user_borrowings.append(borrowing_info)
                        break
        return user_borrowings
    
    def get_overdue_books(self):
        overdue = []
        today = datetime.now()
        
        for record in self.borrowed_books:
            if record['status'] == 'borrowed':
                due_date = datetime.strptime(record['due_date'], "%Y-%m-%d")
                if today > due_date:
                    # Find book and user details
                    for book in self.books:
                        if book['isbn'] == record['isbn']:
                            for user in self.users:
                                if user['user_id'] == record['user_id']:
                                    overdue_info = {
                                        'title': book['title'],
                                        'author': book['author'],
                                        'user_name': user['name'],
                                        'user_id': user['user_id'],
                                        'due_date': record['due_date'],
                                        'days_overdue': (today - due_date).days
                                    }
                                    overdue.append(overdue_info)
                                    break
                            break
        return overdue
    
    def generate_report(self):
        report = {
            'total_books': len(self.books),
            'total_users': len(self.users),
            'currently_borrowed': 0,
            'overdue_books': len(self.get_overdue_books()),
            'available_books': 0
        }
        
        for book in self.books:
            report['available_books'] += book['available']
        
        for record in self.borrowed_books:
            if record['status'] == 'borrowed':
                report['currently_borrowed'] += 1
        
        return report

def main():
    library = LibraryManager()
    
    while True:
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
        
        choice = input("Enter your choice (1-9): ")
        
        if choice == '1':
            title = input("Enter book title: ")
            author = input("Enter author: ")
            isbn = input("Enter ISBN: ")
            quantity = int(input("Enter quantity: "))
            result = library.add_book(title, author, isbn, quantity)
            print(result)
        
        elif choice == '2':
            name = input("Enter user name: ")
            user_id = input("Enter user ID: ")
            email = input("Enter email: ")
            result = library.register_user(name, user_id, email)
            print(result)
        
        elif choice == '3':
            user_id = input("Enter user ID: ")
            isbn = input("Enter ISBN: ")
            days = input("Enter borrow days (default 14): ")
            days = int(days) if days else 14
            result = library.borrow_book(user_id, isbn, days)
            print(result)
        
        elif choice == '4':
            user_id = input("Enter user ID: ")
            isbn = input("Enter ISBN: ")
            result = library.return_book(user_id, isbn)
            print(result)
        
        elif choice == '5':
            search_term = input("Enter search term (title, author, or ISBN): ")
            results = library.find_book(search_term)
            if results:
                for book in results:
                    print(f"Title: {book['title']}, Author: {book['author']}, ISBN: {book['isbn']}, Available: {book['available']}")
            else:
                print("No books found")
        
        elif choice == '6':
            user_id = input("Enter user ID: ")
            borrowings = library.get_user_borrowings(user_id)
            if borrowings:
                for borrowing in borrowings:
                    status = "Overdue" if datetime.now() > datetime.strptime(borrowing['due_date'], "%Y-%m-%d") else "On time"
                    print(f"Title: {borrowing['title']}, Due: {borrowing['due_date']}, Status: {status}")
            else:
                print("No borrowings found")
        
        elif choice == '7':
            overdue = library.get_overdue_books()
            if overdue:
                for book in overdue:
                    print(f"Title: {book['title']}, User: {book['user_name']}, Days overdue: {book['days_overdue']}")
            else:
                print("No overdue books")
        
        elif choice == '8':
            report = library.generate_report()
            print(f"Total Books: {report['total_books']}")
            print(f"Total Users: {report['total_users']}")
            print(f"Currently Borrowed: {report['currently_borrowed']}")
            print(f"Overdue Books: {report['overdue_books']}")
            print(f"Available Books: {report['available_books']}")
        
        elif choice == '9':
            library.save_data()
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
