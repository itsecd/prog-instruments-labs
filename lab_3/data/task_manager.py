import re
import csv
import json
from pathlib import Path

# Регулярные выражения для валидации
PATTERNS = {
    'telephone': r'^(\+7|8)-?\(?\d{3}\)?-?\d{3}-?\d{2}-?\d{2}$',
    'height': r'^\d{1}\.\d{2}$',
    'snils': r'^\d{11}$',
    'identifier': r'^\d{2}-\d{2}/\d{2}$',
    'occupation': r'^[A-Za-zА-Яа-яёЁ\-]+(?: [A-Za-zА-Яа-яёЁ\-]+)*$',
    'longitude': r'^-?\d{1,3}\.\d{5,6}$',  # 5 или 6 знаков после точки
    'blood_type': r'^(A|B|AB|O)[+\u2212\-]$',
    'issn': r'^\d{4}-\d{4}$',
    'locale_code': r'^[a-z]{2}(?:-[a-z]{2})?(?:\.[a-z]{2})?$',
    'date': r'^\d{4}-\d{2}-\d{2}$',
}


def validate_field(value: str, field_name: str) -> bool:
    """Валидация поля по регулярному выражению."""
    pattern = PATTERNS.get(field_name)
    if not pattern:
        return True
    
    # Убираем кавычки и пробелы
    value = value.strip().strip('"')
    
    return bool(re.fullmatch(pattern, value))


def validate_csv(file_path: Path) -> list[int]:
    """Валидация CSV файла и возврат номеров невалидных строк."""
    invalid_rows = []
    
    with open(file_path, 'r', encoding='utf-16') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        for row_num, row in enumerate(reader, start=1):
            is_valid = True
            for field_name, value in row.items():
                if field_name in PATTERNS:
                    if not validate_field(value, field_name):
                        print(f"Row {row_num}, Field '{field_name}': '{value}' is invalid")
                        is_valid = False
                        break
            
            if not is_valid:
                invalid_rows.append(row_num)
    
    return invalid_rows


def calculate_checksum(invalid_rows: list[int], variant: int) -> int:
    """Вычисление контрольной суммы."""
    return sum(invalid_rows) % 1000000 + variant


def main():
    csv_file = Path('84.csv')
    variant = 84
    
    if not csv_file.exists():
        print(f"Файл не найден: {csv_file}")
        return
    
    print(f"Валидация файла {csv_file}...")
    invalid_rows = validate_csv(csv_file)
    
    print(f"\nНайдено {len(invalid_rows)} невалидных строк")
    print(f"Номера строк: {invalid_rows[:20]}...")
    
    checksum = calculate_checksum(invalid_rows, variant)
    print(f"Контрольная сумма: {checksum}")
    
    # Сохранение результата
    result = {
        'variant': variant,
        'invalid_rows_count': len(invalid_rows),
        'checksum': checksum,
        'invalid_rows': invalid_rows
    }
    
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Результат сохранён в result.json")


if __name__ == "__main__":
    main()