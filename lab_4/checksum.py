import json
import hashlib
from pathlib import Path
from typing import List


"""
В этом модуле обитают функции, необходимые для автоматизированной проверки результатов ваших трудов.
"""


RESULT_FILE = Path(__file__).parent / "result.json"


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет md5 хеш от списка целочисленных значений.
    
    Args:
        row_numbers: список номеров строк
    
    Returns:
        str: hex digest MD5 hash
    
    Note:
        Не мутирует входной список
    """
    sorted_numbers = sorted(row_numbers)
    return hashlib.md5(json.dumps(sorted_numbers).encode("utf-8")).hexdigest()


def serialize_result(variant: int, checksum: str, output_file: Path = RESULT_FILE) -> None:
    """
    Метод для сериализации результатов лабораторной.
    
    Args:
        variant: номер варианта
        checksum: вычисленная контрольная сумма
        output_file: путь к выходному файлу (по умолчанию result.json)
    
    Raises:
        IOError: если не удалось записать файл
    """
    result = {"variant": variant, "checksum": checksum}
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
    except IOError as e:
        print(f"Ошибка при записи файла {output_file}: {e}")
        raise


if __name__ == "__main__":
    print(calculate_checksum([1, 2, 3]))
    print(calculate_checksum([3, 2, 1]))
