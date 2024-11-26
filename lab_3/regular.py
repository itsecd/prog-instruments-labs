import csv
import re

import consts


def match_check(string: str, regular: str) -> bool:
    """
    проверяет строку регулярным выражением при помощи re.match

    :param string: строка, которую необходимо проверить
    :param regular: регулярное выражение для этой строки
    :return: True - если значение валидное, False - иначе
    """
    result = re.match(regular, string)
    if not result:
        return False
    return True


def check_csv(file_name: str) -> list[int]:
    """
    Функция ищет в строках .csv файла с именем file_name невалидные значения.
    Возвращает целочисленный список с кол-вом невалидных значений в каждой строке файла

    :param file_name: имя или путь до файла
    :return: целочисленный список с кол-вом невалидных значений в каждой строке
    """
    try:
        with open(file_name, "r", encoding="utf-16") as read_file:
            csv_reader = csv.reader(read_file, delimiter=";")
            headers = next(csv_reader)  # Считываем первую строку как заголовки

            invalid_rows = [
                idx for idx, row in enumerate(csv_reader)
                if any(
                    not match_check(str(value), str(consts.REGULAR_DICT.get(header, "")))
                    for value, header in zip(row, headers)
                )
            ]
        return invalid_rows
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        return []
