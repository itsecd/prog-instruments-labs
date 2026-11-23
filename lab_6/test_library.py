import pytest
import json
from Library_Management_system import Library, main

def test_library_add_book(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    library = Library()
    library.add_book("1", "Python Basics", "John Smith")

    assert "1" in library.books
    assert library.books["1"].title == "Python Basics"
    assert library.books["1"].author == "John Smith"

def test_library_duplicate_book_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    library = Library()
    library.add_book("1", "Python Basics", "John Smith")
    library.add_book("1", "Clean Code", "Robert Martin")

    assert len(library.books) == 1
    assert library.books["1"].title == "Python Basics"

def test_library_issue_book(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    library = Library()
    library.add_book("1", "Python Basics", "John Smith")

    assert library.books["1"].is_issued is False

    library.issue_book("1")

    assert library.books["1"].is_issued is True


def test_library_delete_book_with_mocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    library = Library()
    library.add_book("7", "Clean Code", "Robert Martin")

    save_calls = []

    def fake_save_books():
        save_calls.append(True)
        return True

    monkeypatch.setattr(library, "save_books", fake_save_books)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    library.delete_book("7")

    assert "7" not in library.books
    assert save_calls == [True]


def test_library_return_book(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    library = Library()
    library.add_book("1", "Python Basics", "John Smith")
    library.issue_book("1")

    assert library.books["1"].is_issued is True

    library.return_book("1")

    assert library.books["1"].is_issued is False


def test_library_search_book(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    library = Library()
    library.add_book("1", "Python Basics", "John Smith")
    library.add_book("2", "Clean Code", "Robert Martin")

    capsys.readouterr()

    library.search_book("  PYTHON  ")

    output = capsys.readouterr().out

    assert "Python Basics" in output
    assert "Clean Code" not in output


def test_library_load_books(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data = [
        {
            "book_id": "12",
            "title": "Dune",
            "author": "Frank Herbert",
            "is_issued": True
        }
    ]

    (tmp_path / "books.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    library = Library()

    assert "12" in library.books
    assert library.books["12"].title == "Dune"
    assert library.books["12"].is_issued is True


def test_library_load_corrupted_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "books.json").write_text(
        "{invalid json", encoding="utf-8"
    )

    library = Library()

    output = capsys.readouterr().out

    assert library.books == {}
    assert "corrupted" in output


@pytest.mark.parametrize(
    "book_id, title, author",
    [
        ("abc", "Python", "John Smith"),
        ("1", "", "John Smith"),
        ("1", "Python", "")
    ]
)
def test_library_invalid_book_data(
    tmp_path, monkeypatch, book_id, title, author
):
    monkeypatch.chdir(tmp_path)

    library = Library()
    library.add_book(book_id, title, author)

    assert library.books == {}


def test_main_menu_add_and_search(monkeypatch, capsys):
    class FakeLibrary:
        def __init__(self):
            self.calls = []

        def add_book(self, book_id, title, author):
            self.calls.append(("add", book_id, title, author))

        def search_book(self, keyword):
            self.calls.append(("search", keyword))

    fake_library = FakeLibrary()

    answers = iter([
        "2", "15", "Dune", "Frank Herbert",
        "4", "dune",
        "7"
    ])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("Library_Management_system.Library", lambda: fake_library)

    main()

    assert fake_library.calls == [
        ("add", "15", "Dune", "Frank Herbert"),
        ("search", "dune")
    ]

    assert "Thank you for using Library System." in capsys.readouterr().out