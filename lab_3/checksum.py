import json
import hashlib
from typing import List
import re
import chardet


def detect_encoding(file_path: str) -> str:
    """
    Определяет кодировку файла.

    :param file_path: Путь к файлу.
    :return: Кодировка файла.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Читаем первые 10000 байт
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    return encoding


def is_valid_line(line: str) -> bool:
    """
    Проверяет валидность строки на основе регулярных выражений для каждого поля.

    :param line: Строка, представляющая запись из CSV.
    :return: True, если строка валидна, иначе False.
    """
    fields = line.split(';')
    
    if len(fields) != 10:
        return False

    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    height_pattern = r'^\d+(\.\d+)?$'
    snils_pattern = r'^\d{11}$'
    passport_pattern = r'^\d{2} \d{2} \d{6}$'
    occupation_pattern = r'^[А-Яа-яA-Za-z\s]+$'
    longitude_pattern = r'^-?\d+(\.\d+)?$'
    hex_color_pattern = r'^#[0-9a-fA-F]{6}$'
    issn_pattern = r'^\d{4}-\d{4}$'
    locale_code_pattern = r'^[a-z]{2}-[a-z]{2}$'
    time_pattern = r'^\d{2}:\d{2}:\d{2}\.\d{6}$'

    patterns = [
        email_pattern,
        height_pattern,
        snils_pattern,
        passport_pattern,
        occupation_pattern,
        longitude_pattern,
        hex_color_pattern,
        issn_pattern,
        locale_code_pattern,
        time_pattern
    ]

    for field, pattern in zip(fields, patterns):
        if not re.match(pattern, field.strip('"')):
            return False

    return True


def process_file(input_file: str) -> list[int]:
    """
    Обрабатывает файл и собирает номера невалидных строк.

    :param input_file: Путь к входному CSV файлу.
    :return: Список номеров невалидных строк.
    """
    invalid_lines = []
    encoding = detect_encoding(input_file)
    with open(input_file, 'r', encoding=encoding) as file:
        for line_number, line in enumerate(file, start=1):
            if not is_valid_line(line.strip()):
                invalid_lines.append(line_number)

    return invalid_lines


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.
    :param row_numbers: список целочисленных номеров строк csv-файла, на которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    Метод для сериализации результатов лабораторной.
    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    """
    result = {
        "variant": variant,
        "checksum": checksum
    }
    with open('result.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)
