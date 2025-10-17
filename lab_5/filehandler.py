import json
from typing import Any


class FileHandler:
    @staticmethod
    def read_data(directory: str, mode: str):
        try:
            with open(directory, mode) as file:
                if directory.endswith(".json"):
                    return json.load(file)
                else:
                    return file.read()
        except FileNotFoundError as fe:
            raise FileNotFoundError(f"File was not found: {fe}")
        except json.JSONDecodeError as jde:
            raise ValueError(f"Error decoding the json file: {jde}")
        except Exception as e:
            raise Exception(f"An error occurred when opening the file {e}")

    @staticmethod
    def save_data(directory: str, data: Any, mode: str) -> None:
        try:
            with open(directory, mode) as file:
                if directory.endswith(".json"):
                    json.dump(data, file, ensure_ascii=False, indent=4)
                else:
                    file.write(data)
        except FileNotFoundError as fe:
            raise FileNotFoundError(f"File was not found: {fe}")
        except Exception as e:
            raise Exception(f"An error occurred when saving the file: {e}")
