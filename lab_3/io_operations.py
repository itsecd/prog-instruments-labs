import json
import pandas as pd


def read_csv(filename: str) -> pd.DataFrame:
    """
    Read a csv file and return a data frame.
    :param filename: The name of the csv file.
    :return: Data frame.
    """
    try:
        with open(filename, mode="r", encoding="utf-16") as file:
            df = pd.read_csv(file, delimiter=";")
            return df
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
        return pd.DataFrame()


def read_json(filename: str) -> dict:
    """
    Read a JSON file and return a dictionary.

    :param filename: The name of the JSON file.
    :return: A dictionary with the data.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON format error in file {filename}: {e}")
        return {}
    except Exception as exc:
        print(f"Error reading JSON: {exc}")
        return {}


def write_json(filename: str, data: dict) -> None:
    """
    Write data to a JSON file.

    :param filename: The name of the file to write.
    :param data: The data to write.
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except PermissionError:
        print(f"No permission to write to file {filename}.")
    except Exception as exc:
        print(f"Error writing JSON: {exc}")
