import json
import hashlib
from typing import List

def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.
    Первая строка с данными csv-файла имеет номер 0.
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()

def serialize_result(variant: int, checksum: str) -> None:
    """
    Сериализует результаты лабораторной работы в файл result.json.
    """
    result_data = {
        "variant": variant,
        "checksum": checksum
    }
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f)