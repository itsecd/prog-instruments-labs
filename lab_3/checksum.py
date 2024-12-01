import re
import json
import hashlib
import csv
from typing import List

CSV_INPUT_PATH = "lab_3/7.csv"
OUTPUT_FILE_PATH = "lab_3/result.json"
PATTERN_MAP = {
    "email": r'^\w+@\w+\.\w+',               # Email строго через поддомены
    "height": r'^[1-2]\.\d{2}$',                                                 # Рост — обязательно с двумя знаками
    "inn": r'^\d{12}$',                                                       # ИНН без кавычек
    "passport": r'^\d{2} \d{2} \d{6}$',                                       # Паспорт — с пробелами
    "occupation": r'[А-Я]+|[A-Z]+',                                                 # Профессия в кавычках
    "latitude": r'^-?[1][1-8][1-9]\.\d+|^-?\d{1,2}\.\d+$',                                       # Широта с 6 цифрами
    "hex_color": r'^#[0-9a-f]{6}$',                                        # Шестнадцатиричный цвет
    "issn": r'^\d{4}-\d{4}$',                                                 # ISSN строгий формат
    "uuid": r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',# UUID строгий формат
    "time": r'^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$'
}

"""
Этот модуль содержит вспомогательные функции для проверки корректности данных и подготовки результата.
"""

def calculate_checksum(indices: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.

    ВНИМАНИЕ, ВАЖНО! Чтобы сумма получилась корректной, считать, что первая строка с данными csv-файла имеет номер 0
    Другими словами: В исходном csv 1я строка - заголовки столбцов, 2я и остальные - данные.
    Соответственно, считаем что у 2 строки файла номер 0, у 3й - номер 1 и так далее.

    Надеюсь, я расписал это максимально подробно.
    Хотя что-то мне подсказывает, что обязательно найдется человек, у которого с этим возникнут проблемы.
    Которому я отвечу, что все написано в докстринге 

    :param row_numbers: список целочисленных номеров строк csv-файла, на которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """
    indices.sort()
    return hashlib.md5(json.dumps(indices).encode('utf-8')).hexdigest()

def serialize_result(variant_number: int, checksum: str) -> None:
    """
    Метод для сериализации результатов лабораторной пишите сами.
    Вам нужно заполнить данными - номером варианта и контрольной суммой - файл, лежащий в папке с лабораторной.
    Файл называется, очевидно, result.json.

    ВНИМАНИЕ, ВАЖНО! На json натравлен github action, который проверяет корректность выполнения лабораторной.
    Так что не перемещайте, не переименовывайте и не изменяйте его структуру, если планируете успешно сдать лабу.

    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    """
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as file:
        result_data = {
            "variant": variant_number,
            "checksum": checksum
        }
        file.write(json.dumps(result_data, indent=4))

def read_csv(file_path: str) -> list:
    """
    Загружает содержимое CSV-файла с разделителем `;` и кодировкой UTF-16.

    :param file_path: путь к CSV-файлу.
    :return: список строк данных, представленных в виде списков значений.
    """
    with open(file_path, 'r', encoding='utf-16') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
        return [row for row in reader]

def validate_row(row: list, patterns: dict) -> bool:
    """
    Проверяет, соответствует ли строка заданным регулярным выражениям для каждого поля.

    :param row: список значений в строке.
    :param patterns: словарь регулярных выражений для проверки.
    :return: True, если все поля проходят проверку, иначе False.
    """
    return all(re.match(pattern, value) for pattern, value in zip(patterns.values(), row))

def find_invalid_indices(data: list, patterns: dict) -> list:
    """
    Находит индексы строк, которые не прошли валидацию по заданным шаблонам.

    :param data: список строк, каждая из которых представлена списком значений.
    :param patterns: словарь регулярных выражений для проверки строк.
    :return: список индексов некорректных строк.
    """
    return [i for i, row in enumerate(data) if not validate_row(row, patterns)]

if __name__ == "__main__":
    csv_data = read_csv(CSV_INPUT_PATH)
    error_indices = find_invalid_indices(csv_data, PATTERN_MAP)
    result_checksum = calculate_checksum(error_indices)
    serialize_result(7, result_checksum)





