import json
import os


def read_txt_file(filename: str) -> str:
    """
    Reads text from a file.

    Parameters:
        filename: The name of the file to read from.

    Returns:
        str: The read text.
    """
    try:
        with open(filename, 'r') as file:
            text = file.read().strip()
        return text
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")



