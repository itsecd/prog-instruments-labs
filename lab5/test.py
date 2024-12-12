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
