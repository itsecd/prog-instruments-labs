import re
from typing import List

def validate_row(row: dict, patterns: dict[str, str]) -> bool:
    """
    Проверяет валидность строки данных на основе регулярных выражений.

    :param row: словарь с данными строки
    :param patterns: словарь с регулярными выражениями для валидации
    :return: True если все поля строки соответствуют регулярным выражениям, иначе False
    """
    for field, pattern in patterns.items():
        value = row.get(field, '').strip()
        if not re.fullmatch(pattern, value):
            return False
    return True

def find_error_rows(rows: List[dict], patterns: dict[str, str]) -> List[int]:
    """
    Находит номера строк с ошибками валидации.

    :param rows: список словарей с данными для проверки
    :param patterns: словарь с регулярными выражениями для валидации полей
    :return: список целочисленных номеров строк (начиная с 0), в которых найдены ошибки валидации
    """
    return [i for i, row in enumerate(rows) if not validate_row(row, patterns)]