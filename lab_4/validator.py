import re
from typing import List, Dict, Optional


class ValidationError(Exception):
    """Исключение для ошибок валидации."""
    pass


class DataValidator:
    """Валидатор данных с расширяемым набором patterns."""
    
    def __init__(self):
        self._patterns = {
            "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            "height": re.compile(r"^(0\.[5-9]\d|1\.\d{2}|2\.[0-4]\d|2\.50)$"),
            "inn": re.compile(r"^\d{12}$"),
            "passport": re.compile(r"^\d{2} \d{2} \d{6}$"),
            "occupation": re.compile(r"^[a-zA-Zа-яА-ЯёЁ\s\-]+$"),
            "latitude": re.compile(r"^-?(90(\.0+)?|[0-8]?\d(\.\d+)?)$"),
            "hex_color": re.compile(r"^#[0-9a-fA-F]{6}$"),
            "issn": re.compile(r"^\d{4}-\d{4}$"),
            "uuid": re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            ),
            "time": re.compile(r"^(0\d|1\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$"),
        }
    
    def add_pattern(self, field_name: str, pattern: re.Pattern) -> None:
        """Добавляет новый pattern для валидации."""
        self._patterns[field_name] = pattern
    
    def validate_field(self, field_name: str, value: str) -> bool:
        """Валидирует одно поле."""
        if field_name not in self._patterns:
            return True
        
        pattern = self._patterns[field_name]
        return bool(pattern.match(value))
    
    def validate_row(self, row: Dict[str, str], headers: List[str]) -> bool:
        """
        Проверяет одну строку данных на соответствие регулярным выражениям.
        
        Args:
            row: словарь с данными строки
            headers: список заголовков для проверки
        
        Returns:
            bool: True если строка валидна, False иначе
        """
        for header in headers:
            value = str(row.get(header, ""))
            if not self.validate_field(header, value):
                return False
        return True


_validator = DataValidator()


def validate_row(row: Dict[str, str], headers: List[str]) -> bool:
    """Функция-обертка для совместимости."""
    return _validator.validate_row(row, headers)