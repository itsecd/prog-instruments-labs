import re

import pandas


def read_csv_file(path_to_file: str) -> list:
    """
    Функция, которая считывает файл с таблицей в data_matrix
    и возвращает эту переменную
    :param path_to_file: Путь до файла .csv
    :return: Матрицу, которая содержит данные таблицы не включая заголовки
    """
    df = pandas.read_csv(path_to_file, sep=';', encoding='utf-16')
    data_matrix = df.values.tolist()
    return data_matrix


def check_value(matrix: list, row: int, column: int, pattern: str) -> bool:
    """

    :param matrix: Матрица, в которой проверяется ячейка
    :param row: Номер строки
    :param column: Номер столбца
    :param pattern: Шаблон, по которому сверяют ячейку
    :return: Валидна ячейка или нет
    """
    value = str(matrix[row][column]).strip()
    return bool(re.match(pattern, value))


def find_invalid_rows_in_table(matrix: list, patterns: list) -> list[int]:
    """
    Функция, которая ищет с помощью массива шаблонов, невалидные строки в таблице
    :param matrix: Матрица, в которой ищут некорректные строки
    :param patterns: Список шаблонов, по которому ищут невалидные строки
    :return: Массив номеров невалидных строк
    """
    invalid_rows = []
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            if not check_value(matrix, i, j, patterns[j]):
                invalid_rows.append(i)
                break
    return invalid_rows
