import re
from pandas import DataFrame
from typing import Dict, List, Pattern


def is_invalid_data(pattern: Pattern, value: str) -> bool:
    """
    Проверяет значение на невалидность по регулярному выражению.

    :param pattern: скомпилированный regex паттерн для проверки
    :param value: значение для проверки (преобразуется в строку)
    :return: True если значение невалидно, False если валидно
    """
    return not bool(pattern.fullmatch(str(value)))


def find_invalid_rows(validation_patterns: Dict[str, str], data: DataFrame) -> List[int]:
    """
    Находит индексы строк с невалидными данными на основе regex паттернов.

    Проверяет каждую колонку DataFrame согласно предоставленным regex паттернам
    и возвращает список индексов строк, содержащих хотя бы одно невалидное значение.

    :param validation_patterns: Словарь с паттернами валидации, 
                               где ключ - название колонки, значение - regex паттерн
    :param data: DataFrame для проверки
    :return: Отсортированный список индексов строк с невалидными данными
    """
    invalid_rows = set()
    compiled_patterns = {
        col_name: re.compile(pattern)
        for col_name, pattern in validation_patterns.items()
    }

    for col_name, pattern in compiled_patterns.items():
        if col_name not in data.columns:
            continue

        for index, value in data[col_name].items():
            if is_invalid_data(pattern, value):
                invalid_rows.add(index)

    print(f"Всего найдено невалидных строк: {len(invalid_rows)}")
    return sorted(invalid_rows)
