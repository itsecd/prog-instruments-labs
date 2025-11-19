import csv
import re
from checksum import calculate_checksum, serialize_result
from consts import *


def load_data(file_path: str, delimiter: str = CSV_DELIMITER) -> list[list[str]]:
    encodings = ['utf-8-sig', 'utf-16', 'windows-1251', 'cp1251', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding, newline="") as file:
                reader = csv.reader(file, delimiter=delimiter)
                data = list(reader)
                print(f"Successfully loaded {len(data)} rows with encoding: {encoding}")
                return data
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue

    raise UnicodeDecodeError(f"Cannot decode file {file_path} with any supported encoding")

def validate_row(row: list[str], patterns: list[re.Pattern]) -> bool:
    if len(row) != len(patterns):
        return False

    for pattern, value in zip(patterns, row):
        value = value.strip()
        if not pattern.fullmatch(value):
            return False
    return True

def find_invalid_rows(data: list[list[str]], patterns: list[re.Pattern]) -> list[int]:
    invalid_rows = []

    for index, row in enumerate(data[1:], start=0):
        if not validate_row(row, patterns):
            invalid_rows.append(index)

    return invalid_rows

