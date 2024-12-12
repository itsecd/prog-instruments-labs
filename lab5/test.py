import pytest
from book_search import Book, Library

@pytest.fixture
def library_setup():
    library = Library()
    library.add_book(Book("Гарри Поттер и философский камень", "Джоан Роулинг", 1997, "Фэнтези"))
    library.add_book(Book("Война и мир", "Лев Толстой", 1869, "Роман"))
    return library

def test_find_book_by_title(library_setup):
    results = library_setup.find_book(title="Гарри Поттер")
    assert len(results) == 1
    assert results[0].title == "Гарри Поттер и философский камень"


def test_find_book_by_author(library_setup):
    results = library_setup.find_book(author="Толстой")
    assert len(results) == 1
    assert results[0].author == "Лев Толстой"

def test_find_book_by_genre(library_setup):
    results = library_setup.find_book(genre="Фэнтези")
    assert len(results) == 1
    assert results[0].genre == "Фэнтези"

@pytest.mark.parametrize("title, expected_count", [
    ("Гарри Поттер", 1),
    ("Война", 1),
    ("Незнакомая книга", 0),
])
def test_find_book_with_parametrization(library_setup, title, expected_count):
    results = library_setup.find_book(title=title)
    assert len(results) == expected_count

@pytest.mark.parametrize("author, expected_count", [
    ("Джоан Роулинг", 1),
    ("Лев Толстой", 1),
    ("Неизвестный автор", 0),
])
def test_find_book_by_author_with_parametrization(library_setup, author, expected_count):
    results = library_setup.find_book(author=author)
    assert len(results) == expected_count

def test_remove_book(library_setup):
    library_setup.remove_book("Гарри Поттер и философский камень")
    results = library_setup.find_book(title="Гарри Поттер")
    assert len(results) == 0
