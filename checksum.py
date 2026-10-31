import hashlib
import json
from pathlib import Path
from typing import List


def calculate_checksum(row_numbers: List[int]) -> str:
    
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    
    result = {
        "variant": str(variant),
        "checksum": checksum,
    }
    result_path = Path(__file__).parent / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))


