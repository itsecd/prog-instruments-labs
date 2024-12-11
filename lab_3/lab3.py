import csv
import re
import json
import chardet
from checksum import calculate_checksum
from typing import List, Dict, Any


validation_patterns = [
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # email
    r'^\d+(\.\d{1,2})?$',                                # height
    r'^\d{10,15}$',                                     # inn
    r'^\d{2} \d{2} \d{6}$',                             # passport
    r'^[А-Яа-яЁё\s]+$',                                 # occupation
    r'^[+-]?([1-9]\d*(\.\d+)?|0(\.\d+)?|\.\d+)$',       # latitude
    r'^[#][0-9A-Fa-f]{6}$',                             # hex_color
    r'^\d{4}-\d{4}$',                                   # issn
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'  # uuid
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
    r'^\d{2}:\d{2}:\d{2}\.\d{6}$'                       # time
]


def detect_encoding(file_path: str) -> str:
    """
    Определяет кодировку файла.

    :param file_path: Путь к файлу, кодировку которого необходимо определить.
    :return: Кодировка файла.
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10001))
    return result['encoding']


def is_valid_row(row: List[str]) -> bool:
    """
    Проверяет, что все поля строки соответствуют заданным шаблонам.

    :param row: Список полей строки для проверки.
    :return: True, если все поля валидны, иначе False.
    """
    return all(
        re.match(pattern, field)
        for pattern, field in zip(validation_patterns, row)
    )


def process_csv(file_path: str) -> List[int]:
    """
    Обрабатывает CSV файл, проверяя каждую строку на валидность,
    и возвращает номера невалидных строк.

    :param file_path: Путь к CSV файлу для обработки.
    :return: Список номеров невалидных строк.
    """
    invalid_rows = []
    encoding = detect_encoding(file_path)

    with open(file_path, mode='r', encoding=encoding) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)  # Пропускаем заголовок
        for row_number, row in enumerate(csv_reader, start=2):
            if not is_valid_row(row):
                invalid_rows.append(row_number)

    return invalid_rows


def write_result_to_json(
    control_sum: Any,
    variant_number: int,
    output_file: str
) -> None:
    """
    Записывает результат в JSON файл.

    :param control_sum: Контрольная сумма, которую необходимо записать.
    :param variant_number: Номер варианта, который необходимо записать.
    :param output_file: Путь к выходному JSON файлу.
    """
    result = {
        "variant": variant_number,
        "checksum": control_sum
    }
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)


def read_from_json(file_path: str) -> Dict[str, Any]:
    """
    Читает данные из JSON файла.

    :param file_path: Путь к JSON файлу для чтения.
    :return: Словарь с данными из JSON файла или пустой словарь при ошибке.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            constants = json.load(json_file)
            return constants
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return {}
    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось декодировать JSON из файла '{file_path}'.")
        return {}


if __name__ == '__main__':
    constants = read_from_json("constants.json")
    file_path = constants.get("file_path")
    output_file = constants.get("output_file")
    variant_number = constants.get("variant_number")

    invalid_rows = process_csv(file_path)
    control_sum = calculate_checksum(invalid_rows)

    write_result_to_json(control_sum, variant_number, output_file)
    print(f"Номера невалидных строк: {invalid_rows}")
    print(f"Контрольная сумма: {control_sum}")
