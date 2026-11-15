import hashlib
import json
from typing import List


def calculate_checksum(row_numbers: List[int]) -> str:
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """Заполняет result.json рядом с этим файлом."""
    result = {"variant": variant, "checksum": checksum}
    result_path = __import__('pathlib').Path(__file__).resolve().parent / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))