import csv
import json
import re

from checksum import serialize_result, calculate_checksum


def read_json(path: str) -> dict:
    """
    A function for reading data from a JSON file and returning a dictionary.

    :param path: the path to the JSON file to read
    :return: Dictionary of data from a JSON file
    """
    try:
        with open(path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file '{path}' was not found")
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {str(e)}")


def read_csv(path: str) -> list:
    """
    Reads a CSV file.

    :param path: Path to the CSV file.
    :return: A list of rows from the file, excluding the header.
    """
    data = []
    try:
        with open(path, "r", newline="", encoding="utf-16") as file:
            read_data = csv.reader(file, delimiter=";")
            for row in read_data:
                data.append(row)
        data.pop(0)
    except Exception as e:
        print(f"Error while reading {path}: {e}")
    return data


def check_invalid_row(row: list, patterns: dict) -> bool:
    """
    Checks the validity of each row against the given patterns.

    :param row: A list representing a row from the CSV file.
    :param patterns: A dictionary of patterns to validate each item in the row.
    :return: True if the row is valid, False otherwise.
    """
    for pattern, item in zip(patterns.keys(), row):
        if not re.search(patterns[pattern], item):
            return False
    return True


def get_no_invalid_data_index(path_csv: str, path_json: str) -> list:
    """
    Finds invalid rows in the data and records their indices.

    :param path_csv: Path to the CSV file (not used in this function).
    :param path_json: Path to the JSON file (not used in this function).
    :return: A list of indices of invalid rows.
    """
    invalid_indices = []
    index = 0
    data = read_csv(path_csv)
    patterns = read_json(path_json)
    for row in data:
        if not check_invalid_row(row, patterns):
            invalid_indices.append(index)
        index += 1
    return invalid_indices


if __name__ == "__main__":
    config = read_json("config.json")
    VARIANT = config["VARIANT"]
    PATH_TO_CSV = config["PATH_TO_CSV"]
    PATTERNS = config["PATTERNS"]
    invalid = get_no_invalid_data_index(PATH_TO_CSV, PATTERNS)
    hash_sum = calculate_checksum(invalid)
    serialize_result(VARIANT, hash_sum)
