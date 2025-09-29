import re

import pandas as pd


def check_validate(string: str, regexp: str) -> bool:
    """
    The function checks if the given string is a valid regular expression
    :param string: string to check
    :param regexp: regular expression
    :return:
    """
    return bool(re.fullmatch(rf"{regexp}", string))


def find_invalid_rows(df: pd.DataFrame, regexps: dict) -> set[int]:
    """
    The function finds all rows with invalid values in the given dataframe
    :param df: table as dataframe
    :param regexps: regular expressions
    :return: set with indexes of invalid rows
    """
    invalid_rows_index = set()
    for index, row in df.iterrows():
        for column in df.columns:
            if not check_validate(row[column], regexps[column]):
                invalid_rows_index.add(index)
    return invalid_rows_index
