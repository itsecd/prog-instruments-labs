import json

from log import logger


def read_text(path: str):
    """The function of reading text from file
    Args:
      path: path to the file
    Returns:
      text from the file
    """
    try:
        with open(path, 'r', encoding='UTF-8') as f:
            text = f.read().lower()
        return text
    except FileNotFoundError:
        logger.info("File not found")
        return "File with data not found"
    except Exception as e:
        logger.info("Error reading file")
        return f"Error reading file: {str(e)}"


def write_text(path: str, text: str):
    """The function of writing information to file
    Args:
      path: path to the file
      text: written text
    """
    try:
        with open(path, 'w', encoding='UTF-8') as f:
            f.write(text)
    except FileNotFoundError:
        logger.info("Incorrect path to the directory")
    except Exception as e:
        logger.info(f"Error writing to file: {str(e)}.")


def read_json(file: str):
    """The function of reading data from a json file
    Args:
      file: path to the file
    Returns:
      parameters from json file
    """
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File with settings not found")
        return
    except Exception as e:
        print(f"Error reading file {str(e)}")
        return

    text_1 = data.get("path_to_text_1")
    key_1 = data.get("path_to_key_1")
    encrypted = data.get("path_to_encryption_1")

    text_2 = data.get("path_to_text_2")
    key_2 = data.get("path_to_key_2")
    decrypted = data.get("path_to_decrypt_2")
    alphabet = data.get("path_to_normal_alphabet_2")

    return text_1, key_1, encrypted, text_2, key_2, decrypted, alphabet
