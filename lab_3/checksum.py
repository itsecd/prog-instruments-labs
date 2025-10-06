import json
import hashlib
from typing import List

"""
В этом модуле обитают функции, необходимые для автоматизированной проверки результатов ваших трудов.
"""


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str, filename: str = "result.json") -> None:
    result = {
        "variant": variant,
        "checksum": checksum
    }

    with open(filename, 'w', encoding = 'utf-8') as file:
        json.dump(result, file, ensure_ascii = False, indent = 2)

