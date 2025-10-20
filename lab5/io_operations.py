import json


def read_file(filename: str) -> str:
    """
    Read the content of a text file.

    :param filename: The name of the file to read.
    :return: The content of the file as a string.
    """
    try:
        with open(filename, "r", encoding="UTF-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return ""
    except PermissionError:
        print(f"No access to file {filename}.")
        return ""
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""


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


def write_json(filename: str, data: dict) -> None:
    """
    Write data to a JSON file.

    :param filename: The name of the file to write.
    :param data: The data to write.
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except PermissionError:
        print(f"No permission to write to file {filename}.")
    except Exception as exc:
        print(f"Error writing JSON: {exc}")
