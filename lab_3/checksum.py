import json
import hashlib
from typing import List


def calculate_checksum(row_numbers: List[int]) -> str:

    """
    Вычисляет md5 хеш от списка целочисленных значений.

    :param row_numbers: список целочисленных номеров строк csv-файла, на которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """

    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str, filename: str) -> None:

    """
    Метод для сериализации результатов

    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    :param filename: файл
    """

    result = {'variant': str(variant), 'checksum': checksum}
    with open(filename, 'w') as file:
        json.dump(result, file)
