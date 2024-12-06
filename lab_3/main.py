import csv
import re
import json
import hashlib
from typing import List

PATTERNS = {
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'height': r'^[1-2]\.\d{2}$',
    'snils': r'^\d{11}$',
    'passport': r'^\d{2} \d{2} \d{6}$',
    'occupation': r'^[A-Za-zА-Яа-я\s\-]+$',
    'longitude': r'^-?(180(\.0+)?|((1[0-7][0-9])|([1-9]?[0-9]))(\.\d+)?)$',
    'hex_color': r'^#[0-9a-fA-F]{6}$',
    'issn': r'^\d{4}-\d{4}$',
    'locale_code': r'^[a-z]{2}(?:-[a-z]{2})?$',
    'time': r'^([01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{1,6}$',
}


def is_valid(row: dict) -> bool:
    """
    Проверяет, соответствует ли строка всем заданным паттернам.

    :param row: строка из файла, представлена как словарь
    :return: True, если все поля соответствуют своим шаблонам; иначе False
    """
    for field, pattern in PATTERNS.items():
        if not re.match(pattern, row[field]):
            return False
    return True

def main(variant: int):
    invalid_rows = []
    invalid_row_numbers = []

    with open(f'{variant}.csv', newline='', encoding='utf-16') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row_number, row in enumerate(reader, start=2):  
            if not is_valid(row):
                invalid_rows.append(row)
                invalid_row_numbers.append(row_number - 2)
    print(len(invalid_row_numbers))
    return invalid_row_numbers