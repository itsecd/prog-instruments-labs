import csv
from typing import List
from validator import validate_row
from checksum import calculate_checksum, serialize_result


def main() -> None:
    csv_file = "31.csv"
    invalid_row_indices: List[int] = []
    encoding_to_use = "utf-16"
    
    try:
        with open(csv_file, 'r', encoding=encoding_to_use) as f:
            reader = csv.DictReader(f, delimiter=';')
            headers = reader.fieldnames or []
            
            for row_idx, row in enumerate(reader):
                if not validate_row(row, headers):
                    invalid_row_indices.append(row_idx)
                    
    except UnicodeDecodeError:
        encoding_to_use = "utf-8-sig"
        with open(csv_file, 'r', encoding=encoding_to_use) as f:
            reader = csv.DictReader(f, delimiter=';')
            headers = reader.fieldnames or []
            
            for row_idx, row in enumerate(reader):
                if not validate_row(row, headers):
                    invalid_row_indices.append(row_idx)

    print(f"Файл прочитан в кодировке: {encoding_to_use}")
    print(f"Найдено невалидных строк: {len(invalid_row_indices)}")
    
    checksum = calculate_checksum(invalid_row_indices)
    print(f"Контрольная сумма: {checksum}")
    
    serialize_result(variant=31, checksum=checksum)
    print("Результат успешно записан в result.json")


if __name__ == "__main__":
    main()
