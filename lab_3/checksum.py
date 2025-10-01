import hashlib
import json
from typing import List

def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.

    Важно! Первая строка с данными csv имеет номер 0 (заголовок не учитывается).
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()

def serialize_result(variant: int, checksum: str, path: str) -> None:
    result = {
        "variant": variant,
        "checksum": checksum
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
