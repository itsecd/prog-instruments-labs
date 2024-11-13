import csv
import json
import re

from checksum import calculate_checksum, serialize_result

from typing import Optional, List
from cfg import REGEX_PATTERNS, PATH_TO_CSV, VARIANT

def transform_csv_to_list(path_to_file: str) -> Optional[List]:
    """
    This function gets the path to the .csv file, 
    and then writes the information from there line by line in the form of a list

    Args:
        path_to_file (str): the source csv-file

    Returns:
        Optional[List]: the converted information
    """    
    list = []
    try:
        with open(path_to_file, mode="r", encoding="utf-16") as file:
            text = csv.reader(file, delimiter=";")
            next(text)
            for row in text:
                list.append(row)
            return list
    except Exception as e:
        print(f"transform_csv_to_list: {str(e)}")

def is_valid_row(patterns: dict, row: list) -> bool:
    for key, data in zip(patterns.keys(), row):
        if not re.match(patterns[key], data):
            return False
    return True

def get_indexes_invalid_rows(patterns: dict, data: list) -> Optional[List]:
    """
    This function gets regular expressions, 
    and then adds indexes of rows from data 
    that do not fit these templates to the list

    Args:
        patterns (dict): regular expressions
        data (list): the list of rows

    Returns:
        Optional[List]: the list of invalid rows' indexes
    """    
    list = []
    for i in range(len(data)):
        if not is_valid_row(patterns, data[i]):
            list.append(i)
    return list

if __name__ == "__main__":
    """
    The getting a checksum with checking the number of invalid rows.
    """    
    list = transform_csv_to_list(PATH_TO_CSV)
    inv = get_indexes_invalid_rows(REGEX_PATTERNS, list)
    print(len(inv))
    sum = calculate_checksum(inv)
    print(sum)
    print("Сколько всего невалидных строк: ", len(inv))
    serialize_result(VARIANT, sum)