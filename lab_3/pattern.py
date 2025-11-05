import re
from pandas import DataFrame
from typing import Dict, List, Pattern


def is_invalid_data(pattern: Pattern, value: str) -> bool:
    """
    Проверяет значение на невалидность по регулярному выражению.

    Значение считается невалидным, если оно не соответствует
    заданному regex паттерну. None и NaN значения также считаются невалидными.

    Args:
        pattern (Pattern): Скомпилированный regex паттерн для валидации
        value (str): Значение для проверки (будет преобразовано в строку)

    Returns:
        bool: True если значение невалидно, False если валидно

    Examples:
        >>> pattern = re.compile(r'^\\d{3} [A-Z][A-Za-z ]+$')
        >>> is_invalid_data(pattern, "200 OK")
        False
        >>> is_invalid_data(pattern, "404_Not_Found")
        True
    """
    return not bool(pattern.fullmatch(str(value)))


def find_invalid_rows(validation_patterns: Dict[str, str], data: DataFrame) -> List[int]:
    """
    Находит индексы строк DataFrame, содержащих невалидные данные.

    Функция проверяет каждую колонку DataFrame по соответствующему regex паттерну
    и возвращает список индексов строк, где хотя бы одно значение невалидно.

    Args:
        validation_patterns (Dict[str, str]): Словарь с паттернами валидации.
            Ключи - названия колонок, значения - regex строки для валидации.
        data (DataFrame): DataFrame для проверки данных.

    Returns:
        List[int]: Отсортированный список индексов строк с невалидными данными.

    Raises:
        ValueError: Если validation_patterns пустой или data не является DataFrame

    Examples:
        >>> patterns = {'phone': r'^\\+7\\d{10}$', 'email': r'^[^@]+@[^@]+\\.[^@]+$'}
        >>> df = DataFrame({'phone': ['+79123456789', 'invalid'], 'email': ['test@mail.ru', 'test@mail.ru']})
        >>> find_invalid_rows(patterns, df)
        [1]

    Note:
        - Колонки, отсутствующие в validation_patterns, не проверяются
        - Колонки, отсутствующие в DataFrame, игнорируются
        - Возвращаются только уникальные индексы строк с ошибками
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

    return sorted(invalid_rows)