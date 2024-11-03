import re

from typing import List, Dict

from work_with_file import read_csv_file, read_json_file


def row_valid(row: List[str], patterns: Dict[str, str]) -> bool:
    """
    Checks if a row matches the specified regular expressions.

    Args:
        row: A list of values representing a row of data.
        patterns: A dictionary where keys are field names and values are regular expressions for validation.

    Returns:
        True if all values in the row match their corresponding regular expressions, otherwise False.
    """
    for val, pattern in zip(row, patterns.values()):
        if not re.match(pattern, val):
            return False
    return True


def check_data(path_csv: str, json_path: str) -> List[int]:
    """
    Validates data in a CSV file and returns the indices of invalid rows.

    Args:
        path_csv: The path to the CSV file.
        json_path: The path to the JSON file containing regular expressions.

    Returns:
        A list of indices of invalid rows.
    """
    data = read_csv_file(path_csv)
    reg_exp = read_json_file(json_path)
    
    if data is None or reg_exp is None:
        return []  
    invalid_rows = []
    
    for i, row in enumerate(data):
        if not row_valid(row, reg_exp):
            invalid_rows.append(i)
    
    return invalid_rows
 