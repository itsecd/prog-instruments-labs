import re
import json

import pandas as pd
from pandas import DataFrame

from typing import List


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



def read_csv(filename: str) -> DataFrame:

    """
    Reads data from .csv
    :param filename: Path to .csv
    :return: data: DataFrame
    """

    df = pd.read_csv(filename, sep=';', encoding='utf-16', header=0)

    return df


def validity_check(data: DataFrame, regular_expressions: str) -> List[int]:

    """
    Checking data for validity and filling list of numbers of rows
    :param data: data
    :param regular_expressions: Path to file with expressions
    :return: list of numbers of rows
    """

    regular_expressions = read_json(regular_expressions)
    results = []

    for col, expr in regular_expressions.items():
        for i in data.index:
            if not bool(re.fullmatch(expr, str(data.loc[i, col]))):
                results.append(i)

    return results
