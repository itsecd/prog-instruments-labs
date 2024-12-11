import csv
import json
import re

from typing import Optional, List
from cfg import PATTERNS, PATH_TO_CSV

def transform_csv_to_list(path_to_file: str) -> Optional[List]:
    list = []
    try:
        with open(path_to_file, "r", encoding="utf-16") as file:
            text = csv.reader(file, delimiter=';')
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
    list = []
    for i in range(len(data)):
        if not is_valid_row(patterns, data[i]):
            list.append(i)
    return list