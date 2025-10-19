from typing import Dict, Pattern
from constants import VALIDATION_PATTERNS


def get_validation_patterns() -> Dict[str, Pattern]:
    """
    Returns a dictionary with regular expressions for validation
    :return:Column names mapped to validation patterns
    """
    return VALIDATION_PATTERNS.copy()


def validate_cell(value: str, pattern: Pattern) -> bool:
    """
    Validating a single cell using a regular expression
    :param value:Cell value to check
    :param pattern:Validation regex pattern
    :return:Validation result
    """
    if not value or not isinstance(value, str):
        return False
    return bool(pattern.fullmatch(value.strip()))
