import re
from typing import List, Dict

PATTERNS = {
    'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    'height': re.compile(r'^(0\.[5-9]\d|1\.\d{2}|2\.[0-4]\d|2\.50)$'),
    'inn': re.compile(r'^\d{12}$'),
    'passport': re.compile(r'^\d{2} \d{2} \d{6}$'),
    'occupation': re.compile(r'^[a-zA-Zа-яА-ЯёЁ\s\-]+$'),
    'latitude': re.compile(r'^-?(90(\.0+)?|[0-8]?\d(\.\d+)?)$'),
    'hex_color': re.compile(r'^#[0-9a-fA-F]{6}$'),
    'issn': re.compile(r'^\d{4}-\d{4}$'),
    'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    'time': re.compile(r'^(0\d|1\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$'),
}


def validate_row(row: Dict[str, str], headers: List[str]) -> bool:
    """
    Проверяет одну строку данных на соответствие регулярным выражениям.
    Возвращает True, если строка валидна, и False, если найдено несоответствие.
    """
    for header in headers:
        if header in PATTERNS:
            value = str(row.get(header, ""))
            if not PATTERNS[header].match(value):
                return False
    return True
