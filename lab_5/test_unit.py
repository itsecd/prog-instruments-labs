import pytest
from db import Database
import unittest
from unittest.mock import MagicMock


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


# Сложный тест 1: параметризованное тестирование с pytest
@pytest.mark.parametrize("name, expected", [
    ("John", True),
    ("Jane", True),
    ("Alice", False),
])
def test_find_data(name, expected):
    db = Database(":memory:")
    db.insert("John Doe", "30", "2023-12-01", "john@gmail.com", "Male", "1234567890", "123 Main St")
    db.insert("Jane Doe", "25", "2022-06-15", "jane@gmail.com", "Female", "0987654321", "456 Elm St")
    assert db.find_data(name) == expected


# Сложный тест 2: использование мока
class TestDatabaseWithMock(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        # Эмулируем метод fetch
        self.db.fetch = MagicMock(return_value=[
            (1, "Mocked User", "35", "2020-01-01", "mocked@example.com", "Non-binary", "5555555555", "Mocked Address")
        ])

    def test_mocked_fetch(self):
        # Проверяем эмуляцию метода fetch
        rows = self.db.fetch()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "Mocked User")