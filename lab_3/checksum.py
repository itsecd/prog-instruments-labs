import hashlib
import json
from pathlib import Path
from typing import List


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет MD5-хеш от отсортированного списка номеров строк.
    Нумерация строк с данными начинается с нуля.
    """
    row_numbers.sort()
    return hashlib.md5(
        json.dumps(row_numbers).encode("utf-8")
    ).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """
    Записывает номер варианта и контрольную сумму в result.json.
    Структура JSON соответствует требованиям лабораторной.
    """
    result_path = Path(__file__).with_name("result.json")

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"variant": str(variant), "checksum": checksum},
            file,
            indent=2
        )
        file.write("\n")


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))
