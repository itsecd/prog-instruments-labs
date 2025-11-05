import datetime
import json
from typing import Dict, List, Optional, Union


class Book:
    """Класс для представления книги в библиотеке"""

    def __init__(self, title: str, author: str, year: int, isbn: str, pages: int):
        self.title = title
        self.author = author
        self.year = year
        self.isbn = isbn
        self.pages = pages
        self.is_borrowed = False
        self.borrower_name = None
        self.borrow_date = None

    def borrow(self, borrower_name: str) -> bool:
        """Выдать книгу читателю"""
        if self.is_borrowed:
            return False

        self.is_borrowed = True
        self.borrower_name = borrower_name
        self.borrow_date = datetime.datetime.now()
        return True

    def return_book(self) -> bool:
        """Вернуть книгу в библиотеку"""
        if not self.is_borrowed:
            return False

        self.is_borrowed = False
        self.borrower_name = None
        self.borrow_date = None
        return True

    def get_info(self) -> Dict[str, Union[str, int, bool]]:
        """Получить информацию о книге"""
        return {
            'title': self.title,
            'author': self.author,
            'year': self.year,
            'isbn': self.isbn,
            'pages': self.pages,
            'is_borrowed': self.is_borrowed,
            'borrower_name': self.borrower_name,
            'borrow_date': str(self.borrow_date) if self.borrow_date else None
        }


class Library:
    """Класс для управления библиотекой"""

    def __init__(self, name: str):
        self.name = name
        self.books: Dict[str, Book] = {}
        self.borrow_history: List[Dict] = []

    def add_book(self, book: Book) -> bool:
        """Добавить книгу в библиотеку"""
        if book.isbn in self.books:
            return False

        self.books[book.isbn] = book
        return True

    def remove_book(self, isbn: str) -> bool:
        """Удалить книгу из библиотеки"""
        if isbn not in self.books:
            return False

        del self.books[isbn]
        return True

    def find_book_by_isbn(self, isbn: str) -> Optional[Book]:
        """Найти книгу по ISBN"""
        return self.books.get(isbn)

    def find_books_by_author(self, author: str) -> List[Book]:
        """Найти книги по автору"""
        return [book for book in self.books.values() if book.author.lower() == author.lower()]

    def find_books_by_title(self, title: str) -> List[Book]:
        """Найти книги по названию"""
        return [book for book in self.books.values() if title.lower() in book.title.lower()]

    def borrow_book(self, isbn: str, borrower_name: str) -> bool:
        """Выдать книгу читателю"""
        book = self.find_book_by_isbn(isbn)
        if not book or book.is_borrowed:
            return False

        success = book.borrow(borrower_name)
        if success:
            self.borrow_history.append({
                'isbn': isbn,
                'borrower_name': borrower_name,
                'borrow_date': datetime.datetime.now(),
                'action': 'borrow'
            })
        return success

    def return_book(self, isbn: str) -> bool:
        """Вернуть книгу в библиотеку"""
        book = self.find_book_by_isbn(isbn)
        if not book or not book.is_borrowed:
            return False

        success = book.return_book()
        if success:
            self.borrow_history.append({
                'isbn': isbn,
                'borrower_name': book.borrower_name,
                'return_date': datetime.datetime.now(),
                'action': 'return'
            })
        return success

    def get_available_books(self) -> List[Book]:
        """Получить список доступных книг"""
        return [book for book in self.books.values() if not book.is_borrowed]

    def get_borrowed_books(self) -> List[Book]:
        """Получить список выданных книг"""
        return [book for book in self.books.values() if book.is_borrowed]

    def get_statistics(self) -> Dict[str, int]:
        """Получить статистику библиотеки"""
        total_books = len(self.books)
        available_books = len(self.get_available_books())
        borrowed_books = len(self.get_borrowed_books())

        return {
            'total_books': total_books,
            'available_books': available_books,
            'borrowed_books': borrowed_books,
            'borrow_operations': len([h for h in self.borrow_history if h['action'] == 'borrow'])
        }


class LibraryManager:
    """Класс для управления несколькими библиотеками"""

    def __init__(self):
        self.libraries: Dict[str, Library] = {}

    def create_library(self, name: str) -> bool:
        """Создать новую библиотеку"""
        if name in self.libraries:
            return False

        self.libraries[name] = Library(name)
        return True

    def get_library(self, name: str) -> Optional[Library]:
        """Получить библиотеку по имени"""
        return self.libraries.get(name)

    def remove_library(self, name: str) -> bool:
        """Удалить библиотеку"""
        if name not in self.libraries:
            return False

        del self.libraries[name]
        return True

    def transfer_book(self, from_library: str, to_library: str, isbn: str) -> bool:
        """Переместить книгу между библиотеками"""
        lib_from = self.get_library(from_library)
        lib_to = self.get_library(to_library)

        if not lib_from or not lib_to:
            return False

        book = lib_from.find_book_by_isbn(isbn)
        if not book or book.is_borrowed:
            return False

        if lib_from.remove_book(isbn) and lib_to.add_book(book):
            return True

        # Если перемещение не удалось, возвращаем книгу обратно
        lib_from.add_book(book)
        return False

    def get_all_books(self) -> List[Book]:
        """Получить все книги из всех библиотек"""
        all_books = []
        for library in self.libraries.values():
            all_books.extend(library.books.values())
        return all_books

    def find_book_in_any_library(self, isbn: str) -> Optional[Book]:
        """Найти книгу в любой библиотеке"""
        for library in self.libraries.values():
            book = library.find_book_by_isbn(isbn)
            if book:
                return book
        return None


class ReportGenerator:
    """Класс для генерации отчетов"""

    @staticmethod
    def generate_library_report(library: Library) -> str:
        """Сгенерировать отчет по библиотеке"""
        stats = library.get_statistics()
        report = []
        report.append(f"Отчет для библиотеки: {library.name}")
        report.append("=" * 50)
        report.append(f"Всего книг: {stats['total_books']}")
        report.append(f"Доступно: {stats['available_books']}")
        report.append(f"Выдано: {stats['borrowed_books']}")
        report.append(f"Операций выдачи: {stats['borrow_operations']}")

        if library.get_borrowed_books():
            report.append("\nВыданные книги:")
            for book in library.get_borrowed_books():
                report.append(f"  - {book.title} ({book.author}) - у {book.borrower_name}")

        return "\n".join(report)

    @staticmethod
    def generate_books_report(books: List[Book]) -> str:
        """Сгенерировать отчет по списку книг"""
        if not books:
            return "Книги не найдены"

        report = []
        report.append("Отчет по книгам")
        report.append("=" * 30)

        for i, book in enumerate(books, 1):
            status = "Выдана" if book.is_borrowed else "Доступна"
            report.append(f"{i}. {book.title} - {book.author} ({book.year}) - {status}")

        return "\n".join(report)


def create_sample_library() -> Library:
    """Создать тестовую библиотеку с примерами книг"""
    library = Library("Центральная городская библиотека")

    # Добавляем несколько книг
    books_data = [
        {"title": "Мастер и Маргарита", "author": "Михаил Булгаков", "year": 1967, "isbn": "978-5-699-12345-1", "pages": 480},
        {"title": "Преступление и наказание", "author": "Федор Достоевский", "year": 1866, "isbn": "978-5-699-12345-2", "pages": 672},
        {"title": "Война и мир", "author": "Лев Толстой", "year": 1869, "isbn": "978-5-699-12345-3", "pages": 1225},
        {"title": "1984", "author": "Джордж Оруэлл", "year": 1949, "isbn": "978-5-699-12345-4", "pages": 328},
        {"title": "Гарри Поттер и философский камень", "author": "Джоан Роулинг", "year": 1997, "isbn": "978-5-699-12345-5", "pages": 432},
    ]

    for book_data in books_data:
        book = Book(**book_data)
        library.add_book(book)

    return library


def demonstrate_library_operations():
    """Демонстрация работы библиотеки"""
    print("Демонстрация работы системы управления библиотекой")
    print("=" * 60)

    # Создаем менеджер библиотек
    manager = LibraryManager()

    # Создаем библиотеки
    manager.create_library("Центральная библиотека")
    manager.create_library("Филиал №1")

    central_lib = manager.get_library("Центральная библиотека")
    branch_lib = manager.get_library("Филиал №1")

    # Добавляем книги в центральную библиотеку
    books_to_add = [
        Book("Python для начинающих", "Иван Иванов", 2023, "978-5-12345-678-0", 450),
        Book("Алгоритмы и структуры данных", "Петр Петров", 2022, "978-5-12345-679-7", 600),
        Book("Машинное обучение", "Сидор Сидоров", 2024, "978-5-12345-680-3", 550),
    ]

    for book in books_to_add:
        central_lib.add_book(book)

    # Демонстрируем выдачу книг
    print("\n1. Выдача книг:")
    central_lib.borrow_book("978-5-12345-678-0", "Алексей Алексеев")
    central_lib.borrow_book("978-5-12345-679-7", "Мария Мариева")

    # Показываем доступные и выданные книги
    print("\n2. Доступные книги:")
    available_books = central_lib.get_available_books()
    for book in available_books:
        print(f"  - {book.title}")

    print("\n3. Выданные книги:")
    borrowed_books = central_lib.get_borrowed_books()
    for book in borrowed_books:
        print(f"  - {book.title} - у {book.borrower_name}")

    # Демонстрируем поиск
    print("\n4. Поиск книг по автору 'Иван Иванов':")
    found_books = central_lib.find_books_by_author("Иван Иванов")
    for book in found_books:
        print(f"  - {book.title}")

    # Демонстрируем перемещение книги
    print("\n5. Перемещение книги между библиотеками:")
    success = manager.transfer_book("Центральная библиотека", "Филиал №1", "978-5-12345-680-3")
    print(f"  Перемещение {'успешно' if success else 'не удалось'}")

    # Генерируем отчеты
    print("\n6. Отчет по центральной библиотеке:")
    report = ReportGenerator.generate_library_report(central_lib)
    print(report)

    # Демонстрируем возврат книги
    print("\n7. Возврат книги:")
    central_lib.return_book("978-5-12345-678-0")
    print("  Книга 'Python для начинающих' возвращена")

    # Финальная статистика
    print("\n8. Финальная статистика:")
    final_stats = central_lib.get_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")



def save_library_to_file(library: Library, filename: str) -> bool:
    """Сохранить библиотеку в файл"""
    try:
        data = {
            'name': library.name,
            'books': [book.get_info() for book in library.books.values()],
            'borrow_history': library.borrow_history
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        return True
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")
        return False



def load_library_from_file(filename: str) -> Optional[Library]:
    """Загрузить библиотеку из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        library = Library(data['name'])

        # Восстанавливаем книги
        for book_data in data['books']:
            # Создаем временный словарь без лишних полей
            temp_data = {k: v for k, v in book_data.items()
                        if k in ['title', 'author', 'year', 'isbn', 'pages']}

            book = Book(**temp_data)
            book.is_borrowed = book_data['is_borrowed']
            book.borrower_name = book_data['borrower_name']

            # Восстанавливаем дату выдачи
            if book_data['borrow_date']:
                book.borrow_date = datetime.datetime.fromisoformat(book_data['borrow_date'])

            library.books[book.isbn] = book

        # Восстанавливаем историю
        library.borrow_history = data['borrow_history']

        return library
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return None


class AdvancedBookSearch:
    """Класс для расширенного поиска книг"""

    def __init__(self, library_manager: LibraryManager):
        self.library_manager = library_manager

    def search_by_criteria(self, criteria: Dict[str, str]) -> List[Book]:
        """Поиск книг по различным критериям"""
        all_books = self.library_manager.get_all_books()
        results = all_books

        if 'author' in criteria:
            results = [b for b in results if criteria['author'].lower() in b.author.lower()]

        if 'title' in criteria:
            results = [b for b in results if criteria['title'].lower() in b.title.lower()]

        if 'year_from' in criteria:
            results = [b for b in results if b.year >= int(criteria['year_from'])]

        if 'year_to' in criteria:
            results = [b for b in results if b.year <= int(criteria['year_to'])]

        if 'available_only' in criteria and criteria['available_only'].lower() == 'true':
            results = [b for b in results if not b.is_borrowed]

        return results


def main():
    """Основная функция для демонстрации работы"""
    print("Система управления библиотекой")
    print("=" * 40)

    # Создаем и демонстрируем работу библиотеки
    demonstrate_library_operations()

    # Демонстрация работы с файлами
    print("\n" + "=" * 60)
    print("Демонстрация работы с файлами:")

    sample_library = create_sample_library()

    # Сохраняем библиотеку в файл
    if save_library_to_file(sample_library, "sample_library.json"):
        print("Библиотека сохранена в файл 'sample_library.json'")

    # Загружаем библиотеку из файла
    loaded_library = load_library_from_file("sample_library.json")
    if loaded_library:
        print(f"Библиотека загружена: {loaded_library.name}")
        print(f"Количество книг: {len(loaded_library.books)}")

    # Демонстрация расширенного поиска
    print("\n" + "=" * 60)
    print("Демонстрация расширенного поиска:")

    manager = LibraryManager()
    manager.create_library("Поисковая библиотека")
    search_lib = manager.get_library("Поисковая библиотека")

    # Добавляем разнообразные книги для демонстрации поиска
    diverse_books = [
        Book("Python Advanced", "John Smith", 2020, "111-1111111111", 500),
        Book("Python Basics", "John Smith", 2018, "111-1111111112", 300),
        Book("Data Science", "Alice Johnson", 2022, "111-1111111113", 450),
        Book("Web Development", "Bob Brown", 2019, "111-1111111114", 400),
    ]

    for book in diverse_books:
        search_lib.add_book(book)

    searcher = AdvancedBookSearch(manager)

    # Поиск по автору
    print("\nПоиск книг автора 'John Smith':")
    results = searcher.search_by_criteria({'author': 'John Smith'})
    for book in results:
        print(f"  - {book.title} ({book.year})")

    # Поиск по диапазону лет
    print("\nПоиск книг с 2019 по 2021 год:")
    results = searcher.search_by_criteria({'year_from': '2019', 'year_to': '2021'})
    for book in results:
        print(f"  - {book.title} ({book.year})")


if __name__ == "__main__":
    main()