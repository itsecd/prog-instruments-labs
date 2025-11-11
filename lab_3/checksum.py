import json
import hashlib
from typing import List

def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()

def serialize_result(variant: int, checksum: str) -> None:
    """
    Метод для сериализации результатов лабораторной
    """
    result = {
        "variant": variant,
        "checksum": checksum
    }
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    pass