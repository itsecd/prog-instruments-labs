import re
import csv
from typing import Optional, List


def csv_to_list(file_path: str) -> Optional[List[List[str]]]:
    """
    Читает CSV файл и возвращает список строк, исключая первую строку.

    Args:
        file_path (str): Путь к CSV файлу.

    Returns:
        Optional[List[List[str]]]: Список строк (каждая строка представлена как список),
        или None в случае ошибки.
    """
    try:
        with open(file_path, mode="r", encoding="utf-16") as file:
            reader = csv.reader(file, delimiter=';')
            next(reader)  # Пропускаем первую строку
            return [row for row in reader]
    except Exception as e:
        print(f"Ошибка при чтении CSV файла: {str(e)}.")
        return None


def is_valid_row(pattern: dict, row: List[str]) -> bool:
    """
    Проверяет строку на соответствие регулярным выражениям.

    Args:
        pattern (dict): Словарь с регулярными выражениями для проверки.
        row (List[str]): Строка для проверки.

    Returns:
        bool: True, если строка соответствует всем шаблонам, иначе False.
    """
    return all(re.match(pattern[key], data) for key, data in zip(pattern.keys(), row))


def get_invalid_indices(pattern: dict, data: List[List[str]]) -> Optional[List[int]]:
    """
    Проверяет все строки и возвращает список индексов некорректных строк.

    Args:
        pattern (dict): Регулярные выражения для проверки.
        data (List[List[str]]): Список строк для проверки.

    Returns:
        Optional[List[int]]: Список индексов некорректных строк, или None в случае ошибки.
    """
    invalid_indices = [i for i, row in enumerate(data) if not is_valid_row(pattern, row)]
    return invalid_indices
