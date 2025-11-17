import csv
import json


def read_json(path: str) -> dict:
    """
    Function, which read json file.

    :path: path to json file

    :return: dictionary with data from json file
    """
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def read_csv(path: str) -> list[dict]:
    """
    Function, which read csv file

    :path: path to csv file

    :return: dictionaries' list with data from csv file
    """
    with open(path, 'r', encoding='utf-16', newline='') as file:
        dictionaries = csv.DictReader(file, delimiter=';')
        return list(dictionaries)


def write_json(data: dict, path: str) -> None:
    """
    Write data into json file

    :data: data
    :path: path to json file
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)