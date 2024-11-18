import json
import hashlib
import csv
import re
import chardet
from typing import Dict, List, Any, Union


PATTERNS: Dict[str, str] = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "height": r"^(?:0|1|2)\.\d{2}$",
    "inn": r"^\d{12}$",
    "passport": r"^\d{2} \d{2} \d{6}$",
    "occupation": r"^[a-zA-Zа-яА-ЯёЁ\s-]+$",
    "latitude": r"^(-?[1-8]?\d(?:\.\d{1,})?|90(?:\.0{1,})?)$",
    "hex_color": r"^#[0-9a-fA-F]{6}$",
    "issn": r"^\d{4}-\d{4}$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$",
    "time": r"^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.(\d{1,6})$"
}


def check_patterns(row: List[str], row_number: int) -> bool:
    """
    Проверяет строку на соответствие заданным паттернам.

    :param row: Список значений строки.
    :param row_number: Номер строки в CSV-файле.
    :return: True, если найдена ошибка, иначе False.
    """
    for key, value in enumerate(row):
        field_name = list(PATTERNS.keys())[key]
        if not re.match(PATTERNS[field_name], value):
            print(
                f"Ошибка в строке {row_number}: "
                f"Ошибка в поле '{field_name}': "
                f"значение '{value}' не соответствует паттерну."
            )
            return True
    return False


def detect_encoding(file_path: str) -> str:
    """Определяет кодировку файла.

    Args:
        file_path (str): Путь к файлу.

    Returns:
        str: Кодировка файла.
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    except Exception as e:
        raise RuntimeError(f"Ошибка при определении кодировки файла: {e}")


def process_csv(file_path: str) -> List[Dict[str, Union[Dict[str, Any], List[str]]]]:
    """Обрабатывает CSV файл и находит строки с ошибками.

    Args:
        file_path (str): Путь к CSV файлу.

    Returns:
        List[Dict[str, Union[Dict[str, Any], List[str]]]]: Список строк с ошибками.
    """
    encoding = detect_encoding(file_path)
    error_rows = []
    try:
        with open(file_path, mode='r', encoding=encoding) as csvfile:
            reader = csv.DictReader(csvfile)
            for row_number, row in enumerate(reader, start=1): 
                errors = check_patterns(list(row.values()), row_number)
                if errors:
                    error_rows.append({"row": row, "errors": errors})
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке CSV файла: {e}")
    return error_rows


def calculate_checksum(row_numbers: List[int]) -> str:
    """Вычисляет контрольную сумму для списка номеров строк.

    Args:
        row_numbers (List[int]): Список номеров строк.

    Returns:
        str: Контрольная сумма в формате MD5.
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_to_json(variant: int, checksum: str) -> None:
    """
    Сериализует результат в JSON-файл.

    :param variant: Номер варианта.
    :param checksum: Контрольная сумма.
    """
    try:
        with open('lab_3/result.json', 'r', encoding='utf-8') as json_file:
            result_data: Dict[str, Any] = json.load(json_file)
    except FileNotFoundError:
        result_data = {"variant": variant, "checksum": checksum}

    result_data['checksum'] = checksum
    with open('lab_3/result.json', 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    try:
        with open("lab_3/path_csv.json", "r", encoding='utf-8') as options_file:
            options = json.load(options_file)

        error_data = process_csv(options["csv_file_path"])
        checksum = calculate_checksum(error_data)
        variant_number = 52
        serialize_to_json(variant_number, error_data)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
