from Library_Management_system import Library

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
