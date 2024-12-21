import json
from logger_config import setup_logger

logger = setup_logger(__name__)


def read_json(path: str) -> dict:
    """
    A function for reading data from a JSON file and returning a dictionary.

    Parameters
        path: the path to the JSON file to read
    Returns
        Dictionary of data from a JSON file
    """
    logger.info(f"Reading JSON data from: {path}")
    try:
        with open(path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
        logger.info(f"JSON data successfully read from: {path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        print("The file was not found")
    except Exception as e:
        logger.error(f"An error occurred while reading the JSON file: {str(e)}")
        print(f"An error occurred while reading the JSON file: {str(e)}")
        return None


def write_file(path: str, data: str) -> None:
    """
    A function for writing data to a file.

    Parameters
        path: the path to the file to write
        data: data to write to a file
    """
    logger.info(f"Writing data to file: {path}")
    try:
        with open(path, "w", encoding='UTF-8') as file:
            file.write(data)
        logger.info(f"Data successfully written to file: {path}")
        print(f"The data has been successfully written to the file '{path}'.")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        print("The file was not found")
    except Exception as e:
        logger.error(f"An error occurred while writing the file: {str(e)}")
        print(f"An error occurred while writing the file: {str(e)}")


def read_bytes(file_path: str) -> bytes:
    """
    Reads the contents of a file in binary format.

    Parameters
        file_path: The path to the file to be read.
    Returns
        The contents of the file in binary format.
    """
    logger.info(f"Reading binary data from: {file_path}")
    try:
        with open(file_path, "rb") as file:
            data = file.read()
        logger.info(f"Binary data successfully read from: {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        print("The file was not found")
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {str(e)}")
        print(f"An error occurred while reading the file: {str(e)}")
        return b''


def write_bytes_text(file_path: str, bytes_text: bytes) -> None:
    """
    Writes binary data to a file.

    Parameters
        file_path: The path to the file where the data will be written.
        bytes_text: The binary data to be written to the file.
    """
    logger.info(f"Writing binary data to file: {file_path}")
    try:
        with open(file_path, "wb") as file:
            file.write(bytes_text)
        logger.info(f"Binary data successfully written to file: {file_path}")
        print(f"The data has been successfully written to the file '{file_path}'.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        print("The file was not found")
    except Exception as e:
        logger.error(f"An error occurred while writing the file: {str(e)}")
        print(f"An error occurred while writing the file: {str(e)}")
