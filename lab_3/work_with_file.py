import csv

from json import load
from typing import Dict, List, Optional


def read_json_file(file_name: str) -> Optional[Dict]:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_name: The path to the JSON file.

    Returns:
        A dictionary containing the JSON data if the file is read successfully,
        otherwise None.
    """
    try:
        with open(file_name, mode="r", encoding="utf-8") as file:
            data = load(file)
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def read_csv_file(file_name: str) -> Optional[List[List[str]]]:
    """
    Reads a CSV file and returns its contents as a list of rows without a header string.

    Args:
        file_name: The path to the CSV file.

    Returns:
        A list of lists, where each inner list represents a row in the CSV file,
        or None if an error occurs.
    """
    data = []
    try:
        with open(file_name, "r", encoding="utf-16") as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                data.append(row)
            if data:  
                data.pop(0)  
        return data
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {str(e)}.")
        return None
 