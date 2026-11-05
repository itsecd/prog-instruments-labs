import csv
import re
from checksum import calculate_checksum, serialize_result

VARIANT = 29

PATTERNS = [
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    r"^\d{3} [A-Za-z ]+$",
    r"^\d{12}$",
    r"^\d{2} \d{2} \d{6}$",
    r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$",
    r"^-?\d+\.\d+$",
    r"^#[0-9a-fA-F]{6}$",
    r"^(\d{1,3}-)?\d{1,5}-\d{1,5}-\d{1,5}-\d$",
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    r"^\d{2}:\d{2}:\d{2}\.\d{6}$",
]


def is_valid_row(row: list[str]) -> bool:
    if len(row) != len(PATTERNS):
        return False
        
    for value, pattern in zip(row, PATTERNS):
        if not re.fullmatch(pattern, value):
            return False
    return True


def main() -> None:
    invalid_rows = []
    
    with open("data.csv", "r", encoding="utf-16") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        next(reader)
        
        for row_index, row in enumerate(reader):
            if not is_valid_row(row):
                invalid_rows.append(row_index)
                
    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    
    checksum = calculate_checksum(invalid_rows)
    print(f"Контрольная сумма: {checksum}")
    
    serialize_result(VARIANT, checksum)
    print("Результат сохранён в result.json")


if __name__ == "__main__":
    main()
