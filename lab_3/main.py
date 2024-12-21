import csv
import re
from checksum import calculate_checksum, serialize_result
from typing import List, Generator

VARIANT = 73


class DataValidator:
    """A class for verifying the validity of data."""

    PATTERNS = {
        "email": "^[a-z0-9]+(?:[._][a-z0-9]+)*\@[a-z]+(?:\.[a-z]+)+$",
        "http_status_message": "^\\d{3} [A-Za-z ]+$",
        "snils": "^\\d{11}$",
        "passport": "^\d{2}\s\d{2}\s\d{6}$",
        "ip_v4": "^((25[0-5]|(2[0-4]|1\\d|[1-9]|)\\d)\\.?\\b){4}$",
        "longitude": "^\\-?(180|1[0-7][0-9]|\\d{1,2})\\.\\d+$",
        "hex_color": "^\#[0-9a-fA-F]{6}$",
        "isbn": "^\\d+-\\d+-\\d+-\\d+(?:-\\d+)?$",
        "locale_code": "^[a-z]{2,3}(-[a-z]{2})?$",
        "time": "^(2[0-3]|[0-1][0-9]):[0-5][0-9]:[0-5][0-9]\.\d{6}$"
    }

    def is_valid_row(self, row: List[str]) -> bool:
        """Checks whether the string is valid."""
        for pattern_name, item in zip(self.PATTERNS.keys(), row):
            if not re.search(self.PATTERNS[pattern_name], item):
                return False
        return True


def read_csv_file(file_path: str, encodings: List[str]) \
        -> Generator[List[str], None, None]:
    """Reads a CSV file, trying different encodings."""
    for encoding in encodings:
        try:
            with open(file_path, "r", newline="", encoding=encoding) as file:
                csv_reader = csv.reader(file, delimiter=";")
                next(csv_reader)
                for row in csv_reader:
                    yield row
            print(f"File successfully read with encoding: {encoding}")
            return
        except UnicodeDecodeError:
            print(f"Failed to read '{file_path}' with encoding: {encoding}")
            continue


def get_invalid_row_indices(data_rows: Generator[List[str], None, None],
                            validator: DataValidator) -> List[int]:
    """Finds indexes of invalid rows."""
    invalid_index = []
    for index, row in enumerate(data_rows):
        if not validator.is_valid_row(row):
            invalid_index.append(index)
    return invalid_index


if __name__ == "__main__":
    file_path = "73.csv"
    encodings_to_try = ["utf-16", "utf-8", "windows-1251"]
    validator = DataValidator()

    try:
        data_rows = read_csv_file(file_path, encodings_to_try)
        invalid_indices = get_invalid_row_indices(data_rows, validator)
        serialize_result(VARIANT, calculate_checksum(invalid_indices))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")