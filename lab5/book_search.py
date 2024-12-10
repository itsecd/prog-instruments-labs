class Book:
    def __init__(self, title, author, year, genre):
        self.title = title
        self.author = author
        self.year = year
        self.genre = genre


class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def remove_book(self, title):
        for book in self.books:
            if book.title == title:
                self.books.remove(book)
                return True
        return False

    def find_book(self, title=None, author=None, genre=None):
        results = []
        for book in self.books:
            if (title and title in book.title) or \
                    (author and author in book.author) or \
                    (genre and genre in book.genre):
                results.append(book)
        return results

    def sort_books(self, key):
        return sorted(self.books, key=lambda book: getattr(book, key))

    def filter_books(self, genre=None, year=None):
        results = [
            book for book in self.books
            if (genre and book.genre == genre) or (year and book.year == year)
        ]
        return results

    def search_by_user_input(self):
        title = input("Введите название книги (или нажмите Enter для пропуска): ")
        author = input("Введите имя автора (или нажмите Enter для пропуска): ")
        genre = input("Введите жанр (или нажмите Enter для пропуска): ")

        results = self.find_book(title if title else None, author if author else None, genre if genre else None)

        if results:
            print("Найденные книги:")
            for book in results:
                print(f"Название: {book.title}, Автор: {book.author}, Год: {book.year}, Жанр: {book.genre}")
        else:
            print("Книги не найдены.")


if __name__ == "__main__":
    library = Library()

    library.add_book(Book("Гарри Поттер и философский камень", "Джоан Роулинг", 1997, "Фэнтези"))
    library.add_book(Book("Война и мир", "Лев Толстой", 1869, "Роман"))
    library.add_book(Book("Золотой ключик, или Приключения Буратино", "Алексей Толстой", 1936,
                          "Повесть"))
    library.add_book(Book("Преступление и наказание", "Федор Достоевский", 1869, "Роман"))
    library.add_book(Book("Майор Гром", "Артем Габрелянов", 2012, "Комикс"))
    library.add_book(Book("Тетрадь смерти", "Цугуми Оба", 2003, "Манга"))
    library.add_book(Book("Будни учителя", "Павел Астапов", 2019, "Сборник историй"))
    library.add_book(Book("Граф Монте-Кристо", "Александр Дюма", 1844, "Роман"))
    library.add_book(Book("Мастер и Маргарита", "Михаил Булгаков", 1928, "Роман"))
    library.add_book(Book("Академия проклятий", "Елена Звездная", 2014, "Фэнтези"))
    library.add_book(Book("Bj Алекс", "Минг Гва", 2017, "Манга"))
    library.add_book(Book("Библиография Шерлока Холмса", "Артур Конан Дойл", 1887, "Повесть"))
    library.add_book(Book("Москва слезам не верит", "Валентин Черных", 1994, "Роман"))

    library.search_by_user_input()

    print("Список книг:")
    sorted_books = library.sort_books('year')
    for book in sorted_books:
        print(f"Книга: {book.title}, Год: {book.year}")

