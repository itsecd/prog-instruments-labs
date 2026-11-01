import csv
from pathlib import Path
from typing import List, Dict

from validator import validate_row
from checksum import calculate_checksum, serialize_result


CSV_FILE = Path(__file__).parent / "31.csv"
ENCODINGS_TO_TRY = ["utf-16", "utf-8-sig"]
CSV_DELIMITER = ";"


def read_csv_with_encoding_detection(file_path: Path) -> tuple[Dict, List[Dict]]:
    """
    Читает CSV файл с автоматическим определением кодировки.
    
    Returns:
        tuple: (headers, rows)
    """
    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
                headers = reader.fieldnames or []
                rows = list(reader)
                print(f"Файл прочитан в кодировке: {encoding}")
                return headers, rows
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Не удалось прочитать файл в поддерживаемых кодировках: {ENCODINGS_TO_TRY}")


def find_invalid_rows(rows: List[Dict], headers: List[str]) -> List[int]:
    """Находит индексы строк, не прошедших валидацию."""
    invalid_indices = []
    for row_idx, row in enumerate(rows):
        if not validate_row(row, headers):
            invalid_indices.append(row_idx)
    return invalid_indices


def main() -> None:
    """Основная функция обработки данных."""
    headers, rows = read_csv_with_encoding_detection(CSV_FILE)
    invalid_row_indices = find_invalid_rows(rows, headers)
    
    print(f"Найдено невалидных строк: {len(invalid_row_indices)}")
    
    checksum = calculate_checksum(invalid_row_indices)
    print(f"Контрольная сумма: {checksum}")
    
    serialize_result(variant=31, checksum=checksum)
    print("Результат успешно записан в result.json")


if __name__ == "__main__":
    main()