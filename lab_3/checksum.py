import json
import hashlib

from typing import List

import consts
import regular


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    :param row_numbers: список целочисленных номеров строк csv-файла,
                        на которых были найдены ошибки валидации
    :return: md5 хеш для проверки через github action
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    :param variant: номер вашего варианта
    :param checksum: контрольная сумма, вычисленная через calculate_checksum()
    """
    to_json = {
        "variant": str(variant),
        "checksum": checksum
    }
    with open("result.json", "w", encoding="utf-8") as write_file:
        write_file.write(json.dumps(to_json))
    pass


if __name__ == "__main__":
    serialize_result(consts.VARIANT,
                     calculate_checksum(
                         regular.check_csv(f"{str(consts.VARIANT)}.csv")))
