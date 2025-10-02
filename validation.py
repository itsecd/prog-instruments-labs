import re

"""
This module checks strings for compliance with regular expressions
"""


def validate_row(row, patterns):
    """
    This function checks that all specified row columns match regular expressions
    :param row: list with string data
    :param patterns: regular expressions
    :return: True, if match/else return false
    """
    for column_name, regex_pattern in patterns.items():
        if column_name not in row:
            return False

        value = row[column_name]
        if not re.match(regex_pattern, str(value)):
            return False

    return True
