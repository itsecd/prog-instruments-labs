"""Module for handling file operations: reading and writing JSON and text files."""
import json
import os


def load_json(filepath: str) -> dict:
    """
    Loads and parses a JSON file.

    Args:
        filepath (str): Path to the .json file.

    Returns:
        dict: Parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
        Exception: If any unexpected error occurs during reading.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file '{filepath}' not found.")

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as fnf:
        raise fnf
    except json.JSONDecodeError as jde:
        raise ValueError(f"Invalid JSON format in '{filepath}': {jde}")
    except Exception as e:
        raise Exception(f"Unexpected error loading JSON from '{filepath}': {e}")


def save_json(filepath: str, data: dict) -> None:
    """
    Saves a dictionary to a JSON file.

    Args:
        filepath (str): Path where the .json file will be saved.
        data (dict): Data to save.

    Returns:
        None

    Raises:
        TypeError: If data is not serializable to JSON.
        Exception: If any unexpected error occurs during writing.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except TypeError as te:
        raise TypeError(f"Data is not JSON serializable: {te}")
    except Exception as e:
        raise Exception(f"Unexpected error saving JSON to '{filepath}': {e}")


def read_text_file(filepath: str) -> str:
    """
    Reads content from a text file.

    Args:
        filepath (str): Path to the text file.

    Returns:
        str: Content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If any unexpected error occurs during reading.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Text file '{filepath}' not found.")

        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as fnf:
        raise fnf
    except Exception as e:
        raise Exception(f"Unexpected error reading text file '{filepath}': {e}")


def save_text_file(filepath: str, content: str) -> None:
    """
    Saves string content to a text file.

    Args:
        filepath (str): Path where the text file will be saved.
        content (str): Text content to save.

    Returns:
        None

    Raises:
        Exception: If any unexpected error occurs during writing.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise Exception(f"Unexpected error saving text to '{filepath}': {e}")