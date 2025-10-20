import csv
import json
import re
from checksum import calculate_checksum


def load_data(file_path: str):
    """Loads CSV and returns a list of strings (list[list[str]])."""
    with open(file_path, "r", encoding="windows-1251", newline="") as file:
        reader = csv.reader(file)
        return list(reader)
