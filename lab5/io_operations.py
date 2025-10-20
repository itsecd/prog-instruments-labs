import json
import logging

from lab5.logger import logger

logger = logging.getLogger("io")


def read_file(filename: str) -> str:
    """
    Read the content of a text file.

    :param filename: The name of the file to read.
    :return: The content of the file as a string.
    """
    logger.info(f"Reading file: {filename}")
    try:
        with open(filename, "r", encoding="UTF-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        print(f"File {filename} not found.")
        return ""
    except PermissionError:
        logger.error(f"No access to file: {filename}")
        print(f"No access to file {filename}.")
        return ""
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        print(f"Error reading file {filename}: {e}")
        return ""


def read_json(filename: str) -> dict:
    """
    Read a JSON file and return a dictionary.

    :param filename: The name of the JSON file.
    :return: A dictionary with the data.
    """
    logger.info(f"Reading JSON file: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {filename}")
        print(f"File {filename} not found.")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON format error in {filename}: {e}")
        print(f"JSON format error in file {filename}: {e}")
        return {}
    except Exception as exc:
        logger.error(f"Error reading JSON {filename}: {exc}")
        print(f"Error reading JSON: {exc}")
        return {}


def write_json(filename: str, data: dict) -> None:
    """
    Write data to a JSON file.

    :param filename: The name of the file to write.
    :param data: The data to write.
    """
    logger.info(f"Writing JSON to: {filename}")
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except PermissionError:
        logger.error(f"No permission to write: {filename}")
        print(f"No permission to write to file {filename}.")
    except Exception as exc:
        logger.error(f"Error writing JSON to {filename}: {exc}")
        print(f"Error writing JSON: {exc}")
