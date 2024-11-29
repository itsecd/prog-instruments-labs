import pytest
from work_with_csv import MyCsv


@pytest.fixture
def create_csv_file(tmp_path):
    """
    Фикстура для создания временного CSV-файла.
    """
    def _create_csv_file(content: str, encoding: str = "utf-16"):
        file_path = tmp_path / "test.csv"
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        return str(file_path)

    return _create_csv_file


def test_my_csv_initialization(create_csv_file):
    """
    Тестирование инициализации класса MyCsv.
    """
    csv_content = "Name;Age;City\nAlice;30;New York\nBob;25;Los Angeles"
    csv_path = create_csv_file(csv_content)

    my_csv = MyCsv(csv_path, names=True, delimiter=";")

    assert my_csv.csv_path == csv_path
    assert my_csv.names == ["Name", "Age", "City"]
    assert my_csv.data == [["Alice", "30", "New York"], ["Bob", "25", "Los Angeles"]]


def test_get_values_from_col(create_csv_file):
    """
    Тестирование метода get_values_from_col.
    """
    csv_content = "Name;Age;City\nAlice;30;New York\nBob;25;Los Angeles"
    csv_path = create_csv_file(csv_content)

    my_csv = MyCsv(csv_path, names=True, delimiter=";")

    assert my_csv.get_values_from_col(0) == ["Alice", "Bob"]
    assert my_csv.get_values_from_col(1) == ["30", "25"]
    assert my_csv.get_values_from_col(2) == ["New York", "Los Angeles"]


def test_invalid_delimiter(create_csv_file):
    """
    Тестирование обработки ошибки при некорректном разделителе.
    """
    csv_content = "Name,Age,City\nAlice,30,New York\nBob,25,Los Angeles"
    csv_path = create_csv_file(csv_content)

    with pytest.raises(Exception, match="delimiter must consist of 1 character"):
        MyCsv(csv_path, names=True, delimiter=";;")


def test_empty_file(create_csv_file):
    """
    Тестирование обработки пустого CSV-файла.
    """
    csv_content = ""
    csv_path = create_csv_file(csv_content)

    my_csv = MyCsv(csv_path=csv_path, names=True, delimiter=";")

    assert my_csv.names == []
    assert my_csv.data == []


def test_missing_column(create_csv_file):
    """
    Тестирование доступа к несуществующему столбцу.
    """
    csv_content = "Name;Age;City\nAlice;30;New York"
    csv_path = create_csv_file(csv_content)

    my_csv = MyCsv(csv_path, names=True, delimiter=";")

    with pytest.raises(IndexError):
        my_csv.get_values_from_col(5)
