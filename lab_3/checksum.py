import json
import hashlib
import re
import csv
from typing import List

"""
В этом модуле обитают функции, необходимые для автоматизированной проверки результатов ваших трудов.
"""


patterns = {
    'telephone': r'^"?\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}"?$',
    'height': r'^"?[1-2](?:\.\d{2})?"?$',
    'snils': r'^"?\d{11}"?$',
    'identifier': r'^"?\d{2}-\d{2}/\d{2}"?$',
    'occupation': r'^"?[а-яА-ЯёЁa-zA-Z\s-]+"?$',
    'longitude': r'^"?-?(180(\.0+)?|1[0-7]\d(\.\d+)?|\d{1,2}(\.\d+)?)"?$',
    'blood_type': r'^"?(A|B|AB|O)[\+\u2212]"?$',
    'issn': r'^"?\d{4}-\d{4}"?$',
    'locale_code': r'^[a-z]{2}-[a-z]{2}$|^[a-z]{2}$',
    'date': r'^"?(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"?$'
}

def validate_row(row, row_number):
    for i, value in enumerate(row):
        field_name = list(patterns.keys())[i]
        if not re.match(patterns[field_name], value):
            print(f"Ошибка в строке {row_number}: ошибка в поле '{field_name}': значение '{value}' не соответствует паттерну.")
            return True
    return False

def process_csv(file_path):
    invalid_rows = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)
        for row_number, row in enumerate(reader):
            if validate_row(row, row_number):
                invalid_rows.append(row_number)
    return invalid_rows

# def calculate_checksum(row_numbers: List[int]) -> str:
#     """
#     Вычисляет md5 хеш от списка целочисленных значений.

#     ВНИМАНИЕ, ВАЖНО! Чтобы сумма получилась корректной, считать, что первая строка с данными csv-файла имеет номер 0
#     Другими словами: В исходном csv 1я строка - заголовки столбцов, 2я и остальные - данные.
#     Соответственно, считаем что у 2 строки файла номер 0, у 3й - номер 1 и так далее.

#     Надеюсь, я расписал это максимально подробно.
#     Хотя что-то мне подсказывает, что обязательно найдется человек, у которого с этим возникнут проблемы.
#     Которому я отвечу, что все написано в докстринге ¯\_(ツ)_/¯

#     :param row_numbers: список целочисленных номеров строк csv-файла, на которых были найдены ошибки валидации
#     :return: md5 хеш для проверки через github action
#     """
#     row_numbers.sort()
#     return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


# def serialize_result(variant: int, checksum: str) -> None:
#     """
#     Метод для сериализации результатов лабораторной пишите сами.
#     Вам нужно заполнить данными - номером варианта и контрольной суммой - файл, лежащий в папке с лабораторной.
#     Файл называется, очевидно, result.json.

#     ВНИМАНИЕ, ВАЖНО! На json натравлен github action, который проверяет корректность выполнения лабораторной.
#     Так что не перемещайте, не переименовывайте и не изменяйте его структуру, если планируете успешно сдать лабу.

#     :param variant: номер вашего варианта
#     :param checksum: контрольная сумма, вычисленная через calculate_checksum()
#     """
#     pass


if __name__ == "__main__":
    csv_file_path = 'lab_3/test.csv'
    invalid_row_numbers = process_csv(csv_file_path)
    # print(calculate_checksum([1, 2, 3]))
    # print(calculate_checksum([3, 2, 1]))
