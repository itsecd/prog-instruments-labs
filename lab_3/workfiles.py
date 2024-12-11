import re
import csv

from typing import Optional, List


def csv_to_list(path: str) -> Optional[List]:
    """Function read csv file and return list of rows,
        except for the first line

    Args:
        path (str): path to csv file

    Returns:
        Optional[List]: list of row
    """
    csv_list = []
    try:
        with open(path, mode="r", encoding="utf-16") as file:
            text = csv.reader(file, delimiter=';')
            next(text)
            for row in text:
                csv_list.append(row)
            return csv_list
    except Exception as e:
        print(f"Error with reading csv file: {str(e)}.")


def is_valid_row(pattern: dict, row: str)-> bool:
    """function check row for regex pattern

    Args:
        pattern (dict): pattern for class of data
        row (str): row for checking

    Returns:
        bool: result of check
    """
    for key, data in zip(pattern.keys(), row):
        if not re.match(pattern[key], data):
            return False
    return True


def get_invalid_list(pattern, data: list)-> Optional[List]:
    """function check all rows and appen to list index of invalid rows

    Args:
        pattern (_type_): regex pattern
        data (list): list of rows for checking

    Returns:
        Optional[List]: list with index of invalid rows
    """
    invalid_list = []
    for i in range(len(data)):
        if not is_valid_row(pattern, data[i]):
            invalid_list.append(i)
    return invalid_list
