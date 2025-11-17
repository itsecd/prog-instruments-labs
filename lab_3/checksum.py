import json
import hashlib
from file_processing import write_json
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


def serialize_result(variant: int, checksum: str, result_path: str) -> None:
    """
    Метод для сериализации результатов.
    :param variant: номер варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    :param result_path: путь к файлу, в который будут записаны результаты
    """
    result = {
        "variant": variant,
        "checksum": checksum
    }
    write_json(result, result_path)


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))
