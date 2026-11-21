import csv
import re
from checksum import calculate_checksum, serialize_result

# Регулярные выражения для 10 столбцов (Вариант 11)
PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "height": r"^[0-2]\.\d{2}$",
    "snils": r"^\d{11}$",
    "passport": r"^\d{2} \d{2} \d{6}$",
    "occupation": r"^[A-Za-zА-Яа-яЁё\s-]+$",
    "longitude": r"^-?(?:180(?:\.0+)?|(?:1[0-7]\d|[1-9]?\d)\.\d+)$",
    "hex_color": r"^#[0-9a-fA-F]{6}$",
    "issn": r"^\d{4}-\d{4}$",
    "locale_code": r"^[a-z]{2}(-[a-z]{2})?$",
    "time": r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$"
}

def validate_row(row: dict) -> list:
    """Проверяет строку CSV и возвращает список невалидных полей"""
    invalid_fields = []
    for field, value in row.items():
        if field in PATTERNS:
            # Если значение не соответствует регулярному выражению
            if not re.match(PATTERNS[field], value):
                invalid_fields.append(field)
    return invalid_fields

def main():
    invalid_rows = []
    
    # Читаем CSV (разделитель - точка с запятой, кодировка utf-8)
    # Убедитесь, что файл 11.csv лежит в той же папке, что и validator.py
    with open("11.csv", "r", encoding="utf-16") as f:
        reader = csv.DictReader(f, delimiter=';')
        
        for idx, row in enumerate(reader):
            # idx начинается с 0 для первой строки данных (без заголовка)
            if validate_row(row):
                invalid_rows.append(idx)
    
    # Считаем контрольную сумму
    checksum = calculate_checksum(invalid_rows)
    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")
    
    # Сохраняем результат в result.json
    # ВАЖНО: Если ваш вариант не 1000, а 11, укажите здесь 11
    serialize_result(variant=11, checksum=checksum)

if __name__ == "__main__":
    main()