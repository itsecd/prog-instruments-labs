import json
import hashlib
import csv
import re
import chardet
from typing import Dict, List, Any, Union

# Паттерны для проверки данных
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

def calculate_checksum(row_numbers: List[int]) -> str:
    """Вычисляет контрольную сумму для списка номеров строк.

    Args:
        row_numbers (List[int]): Список номеров строк.

    Returns:
        str: Контрольная сумма в формате MD5.
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()

def check_patterns(data: Dict[str, Any]) -> List[str]:
    """Проверяет данные на соответствие паттернам.

    Args:
        data (Dict[str, Any]): Данные для проверки.

    Returns:
        List[str]: Список ошибок, если есть несоответствия.
    """
    errors = []
    for key, value in data.items():
        pattern = PATTERNS.get(key)
        if pattern and not re.match(pattern, str(value)):
            errors.append(f"Ошибка в поле '{key}': значение '{value}' не соответствует паттерну.")
    return errors

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
            for row in reader:
                errors = check_patterns(row)
                if errors:
                    error_rows.append({"row": row, "errors": errors})
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке CSV файла: {e}")
    return error_rows

def serialize_to_json(data: List[Dict[str, Any]], json_file: str) -> None:
    """Сериализует данные в JSON файл.

    Args:
        data (List[Dict[str, Any]]): Данные для сериализации.
        json_file (str): Путь к выходному JSON файлу.
    """
    try:
        with open(json_file, mode='w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, ensure_ascii=False, indent=4)
    except Exception as e:
        raise RuntimeError(f"Ошибка при записи в JSON файл: {e}")

if __name__ == "__main__":
    try:
        with open("lab_3/path_csv.json", "r", encoding='utf-8') as options_file:
            options = json.load(options_file)

        error_data = process_csv(options["csv_file_path"])
        json_file = 'result.json'
        serialize_to_json(error_data, json_file)
        print(f"Обработка завершена. Результаты сохранены в '{json_file}'.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
