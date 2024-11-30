import re
import json
import hashlib
import csv
from typing import List

INPUT_FILE = "c:/Users/micra/OneDrive/Рабочий стол/lab_3/26.csv"
RESULT_FILE = "c:/Users/micra/OneDrive/Рабочий стол/lab_3/result.json"
PATTERNS = {
    "telephone": r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$',  # Телефон в формате +7-(XXX)-XXX-XX-XX
    "http_status_message": r'^\d{3} [A-Za-z ]+$',      # HTTP-статус: 3 цифры, пробел, текстовое описание
    "snils": r'^\d{11}$',                              # СНИЛС: 11 цифр подряд
    "identifier": r'^\d{2}-\d{2}/\d{2}$',              # Идентификатор: XX-XX/XX
    "ip_v4": r'^(\d{1,3}\.){3}\d{1,3}$',               # IPv4: XXX.XXX.XXX.XXX
    "longitude": r'^-?(?:1[0-7][0-9]|[1-9]?[0-9]|180)\.\d+$', # Долгота WGS84 (от -180 до 180)
    "blood_type": r'^(A|B|AB|O)[+-]$',                 # Группа крови с резус-фактором
    "isbn": r'^\d{3}-\d-\d{5}-\d{3}-\d$',              # ISBN: 13 символов с разделением через тире
    "locale_code": r'^[a-z]{2}-[a-z]{2}(,[a-z]{2})*$', # Код языка: xx-xx[,xx]
    "date": r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$', # Дата в формате YYYY-MM-DD
}

"""
В этом модуле обитают функции, необходимые для автоматизированной проверки результатов ваших трудов.
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
    with open(RESULT_FILE, 'w', encoding='utf-8') as file:
        result_data = {
            "variant": variant_number,
            "checksum": checksum
        }
        file.write(json.dumps(result_data, indent=4))

def load_csv(file_path):
    """
    Читает данные из CSV-файла, пропуская первую строку с заголовками.
    
    input parameters:
    file_path: полный путь к CSV-файлу.

    return:
    список списков, где каждый вложенный список представляет строку данных.
    """
    with open(file_path, 'r', encoding='utf-16') as csv_file:
        return list(csv.reader(csv_file, delimiter=';'))[1:]

def validate_row(row_data: list, patterns: dict) -> bool:
    """
    Проверяет, соответствуют ли данные строки заданным правилам.

    input parameters:
    row_data: значения из строки данных в виде списка.
    patterns: словарь паттернов для проверки.
    return:
    True, если все элементы строки соответствуют, иначе False.
    """
    return all(re.fullmatch(pattern, field) for pattern, field in zip(patterns.values(), row_data))

def find_invalid_rows(data: list, patterns: dict) -> list:
    """
    Определяет индексы строк, не соответствующих правилам проверки.

    input parameters:
    data: список строк данных из файла.
    patterns: словарь паттернов для проверки строк.
    :return: список индексов строк, которые не прошли проверку.
    """
    return [index for index, record in enumerate(data) if not validate_row(record, patterns)]

if __name__ == "__main__":
    data_rows = load_csv(INPUT_FILE)
    invalid_rows = find_invalid_rows(data_rows, PATTERNS)
    checksum_result = calculate_checksum(invalid_rows)
    serialize_result(26, checksum_result)





