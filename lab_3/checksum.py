import re
import json
import hashlib
import csv
from typing import List

CSV_FILE_PATH = "lab_3/1.csv"
RESULT_PATH = "lab_3/result.json"
REGULAR_PATTERN = {
    "email": r"^\w+(\.\w+)*@\w+(\.\w+)+$",
    "http_status_message": r"^\d{3} [A-Za-z ]+$",
    "snils": r"^\d{11}$",
    "passport": r"^\d{2}\s\d{2}\s\d{6}$",
    "ip_v4": r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[0-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[0-9]?[0-9])$",
    "longitude": r"^\-?(180|1[0-7][0-9]|\d{1,2})\.\d+$",
    "hex_color": r"^#[A-Fa-f0-9]{6}$",
    "isbn": r"^(\d{3}-)?\d{1}-\d{5}-\d{3}-\d{1}$",
    "locale_code": r"^[a-zA-Z]+(-[a-zA-Z]+)*$",
    "time": r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d.\d+$"
}

"""
В этом модуле обитают функции, необходимые для автоматизированной проверки результатов ваших трудов.
"""


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.

    ВНИМАНИЕ, ВАЖНО! Чтобы сумма получилась корректной, считать, что первая строка с данными csv-файла имеет номер 0
    Другими словами: В исходном csv 1я строка - заголовки столбцов, 2я и остальные - данные.
    Соответственно, считаем что у 2 строки файла номер 0, у 3й - номер 1 и так далее.

    Надеюсь, я расписал это максимально подробно.
    Хотя что-то мне подсказывает, что обязательно найдется человек, у которого с этим возникнут проблемы.
    Которому я отвечу, что все написано в докстринге ¯\_(ツ)_/¯

    :param row_numbers: список целочисленных номеров строк csv-файла, на которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    Метод для сериализации результатов лабораторной пишите сами.
    Вам нужно заполнить данными - номером варианта и контрольной суммой - файл, лежащий в папке с лабораторной.
    Файл называется, очевидно, result.json.

    ВНИМАНИЕ, ВАЖНО! На json натравлен github action, который проверяет корректность выполнения лабораторной.
    Так что не перемещайте, не переименовывайте и не изменяйте его структуру, если планируете успешно сдать лабу.

    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    """
    with open(RESULT_PATH, 'w', encoding='utf-8') as file:
        result = {
            "variant" : variant,
            "checksum" : checksum
        }
        file.write(json.dumps(result, indent=4))


def load_csv_data(file_path: str) -> list:
    """
    Загружает и считывает CSV-файл в кодировке UTF-16 с разделителем в виде точки с запятой.
    :param file_path (str): Путь к CSV-файлу для чтения.
    
    :return: list: Список строк, где каждая строка представлена в виде списка значений.
    """
    with open(file_path, 'r', encoding='utf-16') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader) 
        rows = [row for row in reader]
    return rows


def is_row_valid(row: list, pattern_dict: dict) -> bool:
    """
    Проверяет, соответствует ли строка данных ожидаемым шаблонам для каждого поля.
    
    :param row (list): Список значений полей в строке.
    :param pattern_dict (dict): Словарь скомпилированных шаблонов регулярных выражений для каждого поля.
        
    :return: bool: True, если все значения в строке соответствуют ожидаемым шаблонам, в противном случае - False.
    """
    for pattern, value in zip(pattern_dict.values(), row):
        if not re.match(pattern, value):
            return False
    return True


def find_invalid_row_indices(rows: list, pattern_dict: dict) -> list:
    """
    Определяет индексы строк в наборе данных, которые не соответствуют требуемым шаблонам.
    
    :param rows (list): Список строк, где каждая строка представляет собой список значений.
    :param pattern_dict (dict): Словарь скомпилированных шаблонов регулярных выражений для проверки.
        
    :return: list: Список индексов для строк, которые являются недопустимыми на основе проверки шаблона.
    """
    invalid_indices = []
    for index, row in enumerate(rows):
        if not is_row_valid(row, pattern_dict):
            invalid_indices.append(index)
    return invalid_indices


if __name__ == "__main__":
    rows = load_csv_data(CSV_FILE_PATH)
    invalid_indeces = find_invalid_row_indices(rows, REGULAR_PATTERN)
    checksum = calculate_checksum(invalid_indeces)
    serialize_result(1, checksum)
 