import json
from typing import Any

import pandas as pd


def read_data(directory: str) -> str | dict[str, Any]:
    """
    The function reads data from json and txt file
    :param directory: path to file
    :return: data from file
    """
    try:
        with open(directory, mode="r", encoding="utf-8") as file:
            if directory.endswith(".json"):
                return json.load(file)
            else:
                return file.read()
    except FileNotFoundError as fe:
        raise FileNotFoundError(f"File was not found: {fe}")
    except json.JSONDecodeError as jde:
        raise ValueError(f"Error decoding the json file: {jde}")
    except Exception as e:
        raise Exception(f"An error occurred when opening the file {e}")


def save_data(directory: str, data: str | dict) -> None:
    """
    The function save data to txt or json file
    :param directory: path to file
    :param data: data that needs to be saved
    :return: None
    """
    try:
        with open(directory, mode="w", encoding="utf-8") as file:
            if directory.endswith(".json"):
                json.dump(data, file)
            else:
                file.write(data)
    except Exception as e:
        raise Exception(f"An error occurred when saving the file: {e}")


def read_csv(directory: str) -> pd.DataFrame:
    """
    The function reads data from csv file
    :param directory: path to csv file
    :return: data as pandas dataframe
    """
    try:
        with open(directory, mode="r", encoding="utf-16") as file:
            return pd.read_csv(file, sep=";")
    except FileNotFoundError as fe:
        raise FileNotFoundError(f"File was not found: {fe}")
    except Exception as e:
        raise Exception(f"An error occurred when opening the file {e}")
