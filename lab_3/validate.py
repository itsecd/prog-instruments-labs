import re
from typing import List


def validate_row(row: dict, patterns: dict[str, str]) -> bool:
    """
    Check validate rows.

    :row: dictionary with row's data
    :patterns: regular expressions

    :return: True or False
    """
    for field, pattern in patterns.items():
        value = row.get(field, '').strip()
        if not re.fullmatch(pattern, value):
            return False
    return True


def invalid_validation_rows(rows: List[dict], patterns: dict[str, str]) -> List[int]:
    """
    Check numbers invalid rows.
    :rows: data
    :patterns: regular expressions

    :return: number's of invalid rows 
    """
    numbers = [i for i, row in enumerate(rows) if not validate_row(row, patterns)]
    return numbers
