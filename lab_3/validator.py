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
