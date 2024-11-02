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



def define_encoding(file_path: str) -> str:
    """
    Defines the file encoding.

    :param file_path: Path to the file.
    :return: File encoding.
    """
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        return encoding
    except OSError as er: 
        raise OSError(f"Error occured during defining the file encoding {file_path}: {er}")


def process_csv(file_path: str) -> List[int]:
    """
    Processes a CSV file and returns line numbers with errors.

    :param file_path: Path to the CSV file.
    :return: List of line numbers with errors.
    """
    invalid_rows: List[int] = []
    encoding = define_encoding(file_path)
    try:
        with open(file_path, newline='', encoding=encoding) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader)
            for row_number, row in enumerate(reader):
                if check_row(row, row_number):
                    invalid_rows.append(row_number)
    except Exception as exc:
        raise Exception(f"Error occured during processing csv file {file_path}: {exc}")
    return invalid_rows


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
    Serializes result 

    :param variant: your variant number
    :param checksum: checksum calculated via calculate_checksum()
    """
    try:
        with open('lab_3/result.json', 'r', encoding='utf-8') as json_file:
            result_data: Dict[str, Any] = json.load(json_file)
    except OSError as er: 
        raise OSError(f"Error occured during serializing the result: {er}")

    result_data['checksum'] = checksum
    with open('lab_3/result.json', 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    try:
        with open("lab_3/settings.json", "r", encoding='utf-8') as options_file:
            options = json.load(options_file)
        invalid_row_numbers = process_csv(options["csv_file_path"])
        checksum = calculate_checksum(invalid_row_numbers)
        variant_number = 54
        serialize_result(variant_number, checksum)

    except OSError as er: 
        raise OSError(f"Error occured during in main section: {er}")
