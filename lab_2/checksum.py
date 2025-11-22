import json
import hashlib
from typing import List


def calculate_checksum(row_numbers: List[int]) -> str:
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    result = {
        "variant": variant,
        "checksum": checksum
    }

    with open("result.json", "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))