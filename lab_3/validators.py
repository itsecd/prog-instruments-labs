import re
from typing import Dict, Pattern
from constants import *


def get_validation_patterns() -> Dict[str, Pattern]:
    """
    Returns a dictionary with regular expressions for validation
    :return:Column names mapped to validation patterns
    """
    patterns = {}
    patterns[COLUMN_EMAIL] = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    patterns[COLUMN_HTTP_STATUS] = re.compile(
        r'^\d{3} [A-Z][A-Za-z ]+[A-Za-z]$'
    )
    patterns[COLUMN_IP_V4] = re.compile(
        r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    )
    patterns[COLUMN_HEX_COLOR] = re.compile(
        r'^#[0-9A-Fa-f]{6}$'
    )
    patterns[COLUMN_INN] = re.compile(
        r'^\d{12}$'
    )
    patterns[COLUMN_PASSPORT] = re.compile(
        r'^\d{2} \d{2} \d{6}$'
    )
    patterns[COLUMN_UUID] = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    remaining_columns = [
        col for col in COLUMNS_TO_VALIDATE
        if col not in patterns
    ]
    for column in remaining_columns:
        patterns[column] = re.compile(r'.*')
    return patterns


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
