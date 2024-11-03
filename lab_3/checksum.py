import json
import hashlib
from typing import List

"""
В этом модуле обитают функции, необходимые для автоматизированной проверки результатов ваших трудов.
"""


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.

    :param row_numbers: список целочисленных номеров строк csv-файла, на которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    Метод для сериализации результатов лабораторной пишите сами.
    Вам нужно заполнить данными - номером варианта и контрольной суммой - файл, лежащий в папке с лабораторной.
    Файл называется, очевидно, result.json.

    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    """
    res = {
        "variant": str(variant), 
        "checksum": checksum
    }

    with open("result.json", mode="w", encoding="utf-8") as f:
        json.dump(res, f)
 