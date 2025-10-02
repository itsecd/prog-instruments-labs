import csv
import json
from json import JSONDecodeError


def csv_open(path):
    """
    This function open csv files
    :param path: path to csv file
    :return: list(csv data)
    """
    try:
        with open(path, "r", encoding="utf-16", newline="") as f:
            text = csv.DictReader(f, delimiter=";")
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


def json_open(path):
    """
    This function open json files
    :param path: path to json file
    :return: json data
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found from path '{path}'")
    except JSONDecodeError as e:
        raise JSONDecodeError(f"Invalid JSON in file '{path}': {e.msg}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"Unexpected error in '{path}': '{e}'")


def json_save(path, data):
    """
    This function save data to json file
    :param path: path to save file
    :param data: information
    :return: None
    """
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        raise Exception(f"Failed to write a file: {e}")
