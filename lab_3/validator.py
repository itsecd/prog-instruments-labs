import re
from patterns import PATTERNS

"""
Функции для валидации данных
"""


def validate_field(field_name: str, value: str) -> bool:
    """Валидация одного поля по регулярному выражению"""
    if field_name not in PATTERNS:
        return True
    pattern = PATTERNS[field_name]
    return bool(re.match(pattern, value))


def validate_row(row: dict) -> bool:
    """Валидация всей строки"""
    for field, value in row.items():
        if not validate_field(field, value):
            return False
    return True


def process_csv_data(lines: list) -> list:
    """Обрабатывает данные CSV и возвращает номера невалидных строк"""
    invalid_rows = []

    for i, line in enumerate(lines[1:], 0):
        fields = line.strip().split(';')

        if len(fields) < 10:
            invalid_rows.append(i)
            continue

        row_data = {
            'telephone': fields[0].strip('"'),
            'height': fields[1].strip('"'),
            'inn': fields[2].strip('"'),
            'identifier': fields[3].strip('"'),
            'occupation': fields[4].strip('"'),
            'latitude': fields[5].strip('"'),
            'blood_type': fields[6].strip('"'),
            'issn': fields[7].strip('"'),
            'uuid': fields[8].strip('"'),
            'date': fields[9].strip('"')
        }

        if not validate_row(row_data):
            invalid_rows.append(i)

    return invalid_rows
