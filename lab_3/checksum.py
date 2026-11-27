"""Расчёт контрольной суммы и сериализация результата лабораторной."""

import hashlib
import json
from pathlib import Path


def calculate_checksum(row_numbers: list[int]) -> str:
    """Вычисляет MD5 от отсортированных индексов строк (индексация от нуля)."""
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode("utf-8")).hexdigest()


def serialize_result(variant: int, checksum: str) -> None:
    """Записывает результат в lab_3/result.json в формате исходного шаблона."""
    result_path = Path(__file__).resolve().with_name("result.json")
    result = {"variant": str(variant), "checksum": checksum}
    with result_path.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")
