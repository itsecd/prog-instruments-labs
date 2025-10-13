import csv
from json import load, dump

def open_json(filename: str) -> dict:
    """
    Read JSON file from the given directory
    :param filename: path to JSON file
    :return: JSON file as a dictionary
    """
    try:
        with open(filename, "r", encoding = "utf-8") as file:
            return load(file)
    except FileNotFoundError as e:
        print(f"File doesn't exist or the path specified is invalid: {e}")
    except PermissionError as e:
        print(f"Can't access this file: {e}")
    except Exception as e:
        print(f"Error reading file: {e}. Check for correct data")


def write_json(filename: str, dictionary: dict) -> None:
    try:
        with open(filename, "w", encoding = "utf-8") as file:
            dump(dictionary, file, ensure_ascii = False, indent = 4)
    except Exception as e:
        print(f"Something went wrong: {e}")

def csv_open(path) -> list:
    """
    Open csv file
    :param path: path to csv file
    :return: list of csv data
    """
    try:
        with open(path, "r", encoding = "utf-16", newline = "") as f:
            text = csv.DictReader(f, delimiter = ";")
            return list(text)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found from path '{path}'")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            "utf-16",
            b"",
            0,
            1,
            f"Failed to decode file '{path}' with encoding utf-16: {e}",
        )
    except csv.Error as e:
        raise csv.Error(f"CSV error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error in '{path}': '{e}'")