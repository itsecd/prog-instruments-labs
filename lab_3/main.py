import csv
import re
from checksum import calculate_checksum, serialize_result

PATTERNS = {
    'telephone': re.compile(r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$'),
    'http_status_message': re.compile(r'^\d{3}\s[A-Za-z\s]+$'),
    'inn': re.compile(r'^\d{12}$'),
    'identifier': re.compile(r'^\d{2}-\d{2}/\d{2}$'),
    'ip_v4': re.compile(r'^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$'),
    'latitude': re.compile(r'^-?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d+)?)$'),
    'blood_type': re.compile(r'^(?:A|B|AB|O)[+\u2212]$'),
    'isbn': re.compile(r'^(?:\d{3}-)?\d-\d{5}-\d{3}-[\dXx]$'),
    'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    'date': re.compile(r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$')
}

def find_invalid_rows(csv_path: str) -> list[int]:
    invalid_row_indices = []
    
    with open(csv_path, mode='r', encoding='utf-16', newline='') as file:
        reader = csv.DictReader(file, delimiter=';')
        for index, row in enumerate(reader):
            for column, pattern in PATTERNS.items():
                value = row.get(column, '')
                if not pattern.fullmatch(value):
                    invalid_row_indices.append(index)
                    break
                    
    return invalid_row_indices

if __name__ == "__main__":
    variant_number = 38
    invalid_rows = find_invalid_rows(f"{variant_number}.csv")
    
    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    checksum_value = calculate_checksum(invalid_rows)
    print(f"Контрольная сумма: {checksum_value}")
    
    serialize_result(variant_number, checksum_value)