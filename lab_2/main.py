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
