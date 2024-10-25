import json
import hashlib
import re
import csv
import chardet
from typing import List

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
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    invalid_rows = []
    try:
        with open(file_path, newline='', encoding=encoding) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader)
            for row_number, row in enumerate(reader):
                if validate_row(row, row_number):
                    invalid_rows.append(row_number)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
    
    return invalid_rows

def calculate_checksum(row_numbers: List[int]) -> str:
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    try:
        with open('lab_3/result.json', 'r', encoding='utf-8') as json_file:
            result_data = json.load(json_file)
    except FileNotFoundError:
        result_data = {"variant": variant, "checksum": checksum}

    result_data['checksum'] = checksum
    
    with open('lab_3/result.json', 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    csv_file_path = 'lab_3/52.csv'
    invalid_row_numbers = process_csv(csv_file_path)
    checksum = calculate_checksum(invalid_row_numbers)
    variant_number = 52
    serialize_result(variant_number, checksum)