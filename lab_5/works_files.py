import json

from enum import Enum


class Mode(Enum):
    frequency = 1
    same_bits = 2
    longest_sequence_in_block = 3


def write_files(path: str, data: str) -> None:
    """
    A function for writing data to a file.

    Parameters
        path: the path to the file to write
        data: data to write to a file
    """
    try:
        with open(path, "a", encoding='UTF-8') as file:
            file.write(data)
        print(f"The data has been successfully written to the file '{path}'.")
    except Exception as e:
        print(f"An error occurred while writing the file: {str(e)}")


def read_json(path: str) -> dict:
    """
    A function for reading data from a JSON file and returning a dictionary.

    Parameters
        path: the path to the JSON file to read
    Returns
        Dictionary of data from a JSON file
    """
    try:
        with open(path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file '{path}' was not found")
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {str(e)}")