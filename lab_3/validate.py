import re

def validate_row(row, regexps):
    """
    Checking string for regular expressions
    :param row: string from csv
    :param regexps: list of regular expressions
    :return: bool
    """
    for col, pattern in regexps.items():
        if col not in row:
            return False
        if not re.match(pattern, row[col]):
            return False
    return True