import argparse
import chardet
import hashlib
import logging
import json
import re

from typing import List

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='lab_3/file.log',
                    filemode='a'
                    )


def detect_encoding(file_path: str) -> str:
    """
    Определяет кодировку файла.

    :param file_path: Путь к файлу.
    :return: Кодировка файла.
    """
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        return encoding
    except Exception as e:
        logging.error(f"Error in determining the encoding of the file: {e}")


def is_valid_line(line: str) -> bool:
    """
    Проверяет валидность строки на основе регулярных выражений для каждого поля.

    :param line: Строка, представляющая запись из CSV.
    :return: True, если строка валидна, иначе False.
    """
    fields = line.split(';')
    
    if len(fields) != 10:
        logging.warning(f"Incorrect number of fields: {len(fields)}.")
        return False

    patterns = [
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # Email
        r'^[0-2]\.\d{2}$',  # Высота
        r'^\d{11}$',  # СНИЛС
        r'^\d{2} \d{2} \d{6}$',  # Паспорт
        r'[a-zA-Zа-яА-ЯёЁ -]+',  # Профессия
        r'^\-?(180|1[0-7][0-9]|\d{1,2})\.\d+$',  # Долгота
        r'^#[A-Fa-f0-9]{6}$',  # Цветовой код
        r'^\d{4}\-\d{4}$',  # ISSN
        r'^[a-zA-Z]+(-[a-zA-Z]+)*$',  # Языковой код
        r'^\d{2}:\d{2}:\d{2}\.\d{6}$'  # Время
    ]

    for field, pattern in zip(fields, patterns):
        if not re.match(pattern, field.strip('"')):
            logging.warning(f"Field '{field.strip()}' does not match the template '{pattern}'. Line: {line.strip()}")
            return False

    return True


def process_file(input_file: str) -> tuple[int, list[int]]:
    """
    Обрабатывает файл и собирает номера невалидных строк.

    :param input_file: Путь к входному CSV файлу.
    :return: Кортеж, содержащий количество невалидных строк и список их номеров.
    """
    try:
        invalid_lines = []
        invalid_count = 0  
        encoding = detect_encoding(input_file)
        with open(input_file, 'r', encoding=encoding) as file:
            next(file)
            for line_number, line in enumerate(file, start=2):
                if not is_valid_line(line.strip()):
                    logging.warning(f"Invalid line {line_number}: {line.strip()}")
                    invalid_lines.append(line_number)
                    invalid_count += 1  

        return invalid_count, invalid_lines
    except Exception as e:
        logging.error(f"Error processing the file: {e}")
        return 0, []


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
    with open('lab_3/result.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Программа для вычисления контрольной суммы')
    parser.add_argument('--input_file',
                        type=str,
                        help='Путь к CSV файлу',
                        default="lab_3/35.csv")
    parser.add_argument('--var',
                        type=int,
                        help='Номер варианта',
                        default=35)
    args = parser.parse_args()
    
    invalid_count, invalid_lines = process_file(args.input_file)
    
    if invalid_count > 0:
        logging.info(f"Processed {invalid_count} invalid lines: {invalid_lines}.")
    else:
        logging.info("All lines is valid.")

    checksum = calculate_checksum(invalid_lines)
    serialize_result(args.var, checksum)