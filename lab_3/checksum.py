import csv
import hashlib
import json
import re
from typing import Any, Dict, List

import chardet


patterns: Dict[str, str] = {
    'telephone': r'',
    'http_status_message': r'',
    'inn': r'',
    'identifier': r'',
    'ip_v4': r'',
    'latitude': r'',
    'blood_type': r'',
    'isbn': r'',
    'uuid': r'',
    'date': r''
}

def check_row(row: List[str], row_number: int) -> bool:
    """
    Checks a string for compliance with the patterns.

    :param row: List of row values.
    :param row_number: Row number in the CSV file.
    :return: True if an error was found, False otherwise.
    """
    for i, value in enumerate(row):
        field_name = list(patterns.keys())[i]
        if not re.match(patterns[field_name], value):
            print(f"Ошибка в строке {row_number}: "
                  f"Ошибка в поле '{field_name}': "
                  f"значение '{value}' не соответствует паттерну.")
            return True
    return False


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Calculates checksum for list of string's number

    :param row_numbers: list of integer row numbers of the csv file where validation errors were found
    :return: md5 hash for checking via github action
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    """
    pass


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))
