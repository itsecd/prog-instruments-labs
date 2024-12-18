import json


def read_json(path_to_file: str) -> dict:
    """
    Read JSON content from a file.

    Args:
        path_to_file (str): Location of the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    try:
        with open(path_to_file, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    except FileNotFoundError as file_error:
        raise FileNotFoundError(f"File {path_to_file} was not found.") from file_error
    except Exception as error:
        raise error


def write_text(path_to_file: str, content: str) -> bool:
    """
    Write given text content to a specified file.

    Args:
        path_to_file (str): Location of the .txt file to write.
        content (str): Text to be saved.

    Returns:
        bool: True if save was successful, False otherwise.
    """
    is_saved = True
    try:
        with open(path_to_file, 'a', encoding='utf-8') as file:
            file.write(content)
    except FileNotFoundError:
        print(f"Saving failed: file {path_to_file} not found.")
        is_saved = False
    except Exception as error:
        print(f"Unexpected error: {error}")
        is_saved = False
    return is_saved