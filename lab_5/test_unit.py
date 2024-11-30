import pytest
from db import Database


@pytest.fixture
def test_data_base() -> Database:
    return Database("AnotherDB.db")


def test_insert():
    data = Database("AnotherDB.db")
    data.insert('Daniel', '30', '2000-01-07', 'daniel@gmail.com', 'Male', '0524567890', '759 Puma St')
    data.insert('Alice', '25', '2022-03-15', 'alice@example.com', 'Female', '9876543210', '456 Oak St')
    data.insert("Lisa", "29", "1999-04-07", "lisa111@gmail.com", "Female", "4424789440", "759 Puma St")
    data.insert("Anton", "27", "1996-04-07", "Anton1@gmail.com", "Male", "4424d9440", "759 Puma St")
    assert len(data.fetch()) == 4


def test_fetch():
    data = Database("AnotherDB.db")
    tmp = data.fetch()
    assert tmp == [(1, 'Daniel', '30', '2000-01-07', 'daniel@gmail.com', 'Male', '0524567890', '759 Puma St'),
                   (2, 'Alice', '25', '2022-03-15', 'alice@example.com', 'Female', '9876543210', '456 Oak St'),
                   (3, "Lisa", "29", "1999-04-07", "lisa111@gmail.com", "Female", "4424789440", "759 Puma St"),
                   (4, "Anton", "27", "1996-04-07", "Anton1@gmail.com", "Male", "4424d9440", "759 Puma St")]


def test_getDataInd():
    data = Database("AnotherDB.db")
    dt = data.get_data_ind(3)
    assert dt == (3, "Lisa", "29", "1999-04-07", "lisa111@gmail.com", "Female", "4424789440", "759 Puma St")


def test_FindName():
    data = Database("AnotherDB.db")
    data_name = data.find_data("Lisa")
    assert data_name is True


def test_remove():
    data = Database("AnotherDB.db")
    data.remove(4)
    tmp = data.fetch()
    assert tmp == [(1, 'Daniel', '30', '2000-01-07', 'daniel@gmail.com', 'Male', '0524567890', '759 Puma St'),
                   (2, 'Alice', '25', '2022-03-15', 'alice@example.com', 'Female', '9876543210', '456 Oak St'),
                   (3, "Lisa", "29", "1999-04-07", "lisa111@gmail.com", "Female", "4424789440", "759 Puma St")]


def test_clearAll():
    data = Database("AnotherDB.db")
    data.clear_ALL()
    assert len(data.fetch()) == 0


def test_dataNull():
    data = Database("AnotherDB.db")
    assert data.data_is_null() is True
