from pandas import DataFrame
import re

"""
модуль для работы с регулярными выражениями
"""


def validate_by_pattern(data: str, pattern: str) -> bool:
    """
    Проверяет подходит ли строка под паттерн
    :param data: строка для проверки
    :param pattern: паттерн в виде регулярного выражения
    :return: Подходит или нет
    """
    sub_pattern = rf'{pattern}'
    if re.fullmatch(sub_pattern, data):
        return True
    return False


def get_rows_with_mistakes(data_frame: DataFrame, patterns) -> list:
    """
    Сравнивает каждое значение (по столбцам) с паттерном(разный для каждого столбца)
    :param data_frame: датафрейи
    :param patterns: список паттернов
    :return: лист с номерами строк с ошибками
    """
    rows_with_mistakes = []
    patter_index = 0
    for column_name in data_frame:
        for row_index, value in data_frame[column_name].items():
            if not validate_by_pattern(value, patterns[patter_index]):
                if row_index not in rows_with_mistakes:
                    rows_with_mistakes.append(row_index)
        patter_index += 1
    return rows_with_mistakes
