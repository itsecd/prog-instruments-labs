import json


def read_text(path: str):
    """ func reads text from file
    Args:
      path: file path
    Returns:
      file's text
    """
    try:
        with open(path, 'r', encoding='UTF-8') as f:
            text = f.read().lower()
        return text
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_text(path: str, text: str):
    """ func writes info in file
    Args:
      path: path to file
      text: text
    """
    try:
        with open(path, 'w', encoding='UTF-8') as f:
            f.write(text)
    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"Error writing to file: {str(e)}.")


def read_json(file: str):
    """ func reads text from json file
    Args:
      file: file path
    Returns:
      parameters out of json
    """
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found")
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
