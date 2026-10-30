import csv
import re
from pathlib import Path
from typing import Dict, List, Pattern, Set, Tuple

from checksum import calculate_checksum, serialize_result

VALIDATION_PATTERNS: Dict[str, Pattern] = {
    'telephone': re.compile(r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$'),
    'height': re.compile(r'^(?:1\.\d{2}|2\.00)$'),
    'inn': re.compile(r'^\d{12}$'),
    'identifier': re.compile(r'^\d{2}-\d{2}/\d{2}$'),
    'occupation': re.compile(r'^[а-яА-ЯёЁa-zA-Z\s\-]+$'),
    'latitude': re.compile(r'^-?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d{1,6})?)$'),
    'blood_type': re.compile(r'^(?:A|B|AB|O)[+\u2212\-]$'),
    'issn': re.compile(r'^\d{4}-\d{4}$'),
    'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    'date': re.compile(r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$'),
}

FIELD_ORDER: List[str] = [
    'telephone', 'height', 'inn', 'identifier', 'occupation',
    'latitude', 'blood_type', 'issn', 'uuid', 'date',
]

CSV_FILE = '40.csv'
VARIANT = 40
ENCODING = 'utf-16'


def validate_height_range(value: str) -> bool:
    """Дополнительная проверка диапазона роста (1.50-2.00)."""
    try:
        return 1.50 <= float(value) <= 2.00
    except ValueError:
        return False


def validate_latitude_range(value: str) -> bool:
    """Дополнительная проверка диапазона широты (-90..90)."""
    try:
        return -90.0 <= float(value) <= 90.0
    except ValueError:
        return False


def validate_date_real(value: str) -> bool:
    """Проверка реалистичности даты (дни в месяце, высокосный год)."""
    try:
        year, month, day = map(int, value.split('-'))
    except ValueError:
        return False
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 1 <= day <= 31
    if month in (4, 6, 9, 11):
        return 1 <= day <= 30
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return 1 <= day <= (29 if is_leap else 28)


def validate_cell(value: str, field_name: str, row_number: int) -> Tuple[bool, str]:
    """Универсальная валидация одной ячейки по правилам поля."""
    value = value.strip()
    pattern = VALIDATION_PATTERNS.get(field_name)

    if pattern is None:
        return True, ""

    if not pattern.match(value):
        return False, f"Row {row_number}, {field_name}: '{value}' не соответствует паттерну"

    if field_name == 'height' and not validate_height_range(value):
        return False, f"Row {row_number}, height: '{value}' вне диапазона 1.50-2.00"

    if field_name == 'latitude' and not validate_latitude_range(value):
        return False, f"Row {row_number}, latitude: '{value}' вне диапазона -90..90"

    if field_name == 'date' and not validate_date_real(value):
        return False, f"Row {row_number}, date: '{value}' - невалидная дата"

    return True, ""


def analyze_csv(file_path: str) -> Dict[str, int]:
    """Предварительный анализ структуры файла: столбцы и строки данных."""
    with open(file_path, 'r', encoding=ENCODING) as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        total_rows = sum(1 for _ in reader)
    return {'columns': len(header), 'total_rows': total_rows}


def validate_csv_file(file_path: str) -> List[int]:
    """Валидация всего файла. Возвращает отсортированные номера невалидных строк."""
    invalid_rows: Set[int] = set()

    with open(file_path, 'r', encoding=ENCODING) as f:
        reader = csv.reader(f, delimiter=';')

        for row_number, row in enumerate(reader, start=1):
            if row_number == 1:  # пропускаем заголовок
                continue

            if len(row) != len(FIELD_ORDER):
                invalid_rows.add(row_number)
                continue

            for value, field_name in zip(row, FIELD_ORDER):
                is_valid, _ = validate_cell(value, field_name, row_number)
                if not is_valid:
                    invalid_rows.add(row_number)
                    break

    return sorted(invalid_rows)


def main() -> None:
    if not Path(CSV_FILE).exists():
        print(f"Ошибка: файл '{CSV_FILE}' не найден.")
        return

    stats = analyze_csv(CSV_FILE)
    print(f"Столбцов: {stats['columns']}, всего строк: {stats['total_rows']}")

    invalid_rows = validate_csv_file(CSV_FILE)
    print(f"Найдено невалидных строк: {len(invalid_rows)}")

    checksum = calculate_checksum(invalid_rows)
    print(f"Контрольная сумма: {checksum}")

    serialize_result(VARIANT, checksum)
    print("Результат сохранен в result.json")


if __name__ == "__main__":
    main()