
from Library_Management_system import Book


def test_book_initialization():
    book = Book("1", "Python Basics", "John Smith")

    assert book.book_id == "1"
    assert book.title == "Python Basics"
    assert book.author == "John Smith"
    assert book.is_issued is False


def test_book_to_dict():
    book = Book("2", "Clean Code", "Robert Martin")

    result = book.to_dict()

    assert result == {
        "book_id": "2",
        "title": "Clean Code",
        "author": "Robert Martin",
        "is_issued": False
    }


def test_book_from_dict():
    data = {
        "book_id": "3",
        "title": "Harry Potter",
        "author": "J. K. Rowling",
        "is_issued": True
    }

    book = Book.from_dict(data)

    assert book.book_id == "3"
    assert book.title == "Harry Potter"
    assert book.author == "J. K. Rowling"
    assert book.is_issued is True


def test_book_str():
    book = Book("4", "Clean Architecture", "Robert Martin")

    result = str(book)

    assert result == "ID: 4 | Clean Architecture by Robert Martin [Available]"