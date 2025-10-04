import re
from pandas import DataFrame
from typing import List, Dict


def validate_by_pattern(data: str, pattern: str) -> bool:
    """
    Проверка подходит ли строка под паттерн
    :param data: строка для проверки
    :param pattern: паттерн в виде регулярного выражения
    :return: bool
    """
    if data is None or data != data:
        return False

    try:
        if re.fullmatch(pattern, str(data)):
            return True
        return False
    except re.error:
        return False


def find_error_rows(df: DataFrame, patterns_dict: Dict[str, str]) -> List[int]:
    """
    Находит номера строк с ошибками, проходя по всем столбцам
    :param df: DataFrame с данными из CSV
    :param patterns_dict: Словарь {название_столбца: паттерн}
    :return: Отсортированный список номеров строк с ошибками
    """
    error_rows = set()

    for column_name, pattern in patterns_dict.items():
        if column_name in df.columns:
            for row_index, value in df[column_name].items():
                if not validate_by_pattern(value, pattern):
                    error_rows.add(row_index)

    return sorted(list(error_rows))