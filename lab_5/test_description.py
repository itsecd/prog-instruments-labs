import os
import csv
import pytest
from tempfile import TemporaryDirectory
from description import about, make_description


@pytest.fixture
def create_temp_files():
    """
    Фикстура для создания временной директории с файлами и папками для тестирования.
    """
    with TemporaryDirectory(dir=os.getcwd()) as temp_dir:
        # Создаем файлы с именами, содержащими тип классов
        file1 = os.path.join(temp_dir, f"{os.path.basename(temp_dir)}_class1_file1.txt")
        file2 = os.path.join(temp_dir, f"{os.path.basename(temp_dir)}_class2_file2.txt")
        file3 = os.path.join(temp_dir, f"{os.path.basename(temp_dir)}_file_without_class.txt")

        # Создаем временные файлы
        with open(file1, "w") as f:
            f.write("Sample content")

        with open(file2, "w") as f:
            f.write("Sample content")

        with open(file3, "w") as f:
            f.write("Sample content")

        # Создаем вложенную папку и файл внутри неё
        nested_dir = os.path.join(temp_dir, "nested")
        os.mkdir(nested_dir)
        nested_file = os.path.join(nested_dir, f"{os.path.basename(nested_dir)}_class3_nestedfile.txt")

        with open(nested_file, "w") as f:
            f.write("Sample content")

        yield temp_dir


def test_about(create_temp_files):
    """
    Тестирование функции about на сбор описаний файлов из директории.
    """
    temp_dir = create_temp_files
    descriptions = about(temp_dir)

    # Проверяем количество файлов в выводе
    assert len(descriptions) == 4

    # Проверяем правильность первого файла
    elem = f"{os.path.basename(temp_dir)}_class1_file1.txt"
    assert descriptions[0].full_path.endswith(f"{os.path.basename(temp_dir)}_class1_file1.txt")
    ind = elem.find("_")
    if not (ind == -1):
        type_class = elem[0:ind]
    assert descriptions[0].type_class == type_class

    # Проверяем файл без типа класса
    ind = os.path.basename(temp_dir).find("_")
    if not (ind == -1):
        type_class = elem[0:ind]
    assert descriptions[2].type_class == type_class

    # Проверяем правильность вложенного файла
    assert descriptions[3].full_path.endswith("nested_class3_nestedfile.txt")
    assert descriptions[3].type_class == "nested"


def test_make_description(create_temp_files):
    """
    Тестирование функции make_description на создание CSV-файла с описанием файлов.
    """
    temp_dir = create_temp_files
    csv_name = "test_description"

    # Вызываем make_description для создания файла CSV
    make_description(csv_name, temp_dir, "txt")

    # Путь к созданному CSV-файлу
    csv_path = os.path.join(temp_dir, csv_name + ".csv")

    # Проверяем, что файл был создан
    assert os.path.exists(csv_path)

    # Проверяем содержимое CSV-файла
    with open(csv_path, encoding='utf-16') as f:
        reader = csv.reader(f, delimiter=",")
        rows = list(reader)

    # Проверяем, что заголовок CSV-файла правильный
    assert rows[0] == ["абсолютный путь к файлу", "относительный путь", "метка класса"]

    # Проверяем количество записей
    assert len(rows) == 5*2  # 4 файла + заголовок + пустые строки

    # Проверяем корректность данных для первого файла
    elem = f"{os.path.basename(temp_dir)}_class1_file1.txt"
    assert rows[2][0].endswith(elem)
    ind = elem.find("_")
    if not (ind == -1):
        type_class = elem[0:ind]
    assert rows[2][2] == type_class
