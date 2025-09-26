import json


"""
Модуль для работы с файлами
"""


def read_json(path: str) -> dict:
    """
    Чтение данных из json файла
    :param path: путь к файл
    :return: словарь с данными
    """
    try:
        with open(path, mode="r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as not_found:
        raise FileNotFoundError(f"File was not found: {not_found}")
    except json.JSONDecodeError as decode_error:
        raise ValueError(f"Error decoded the json file: {decode_error}")
    except Exception as exc:
        raise Exception(f"An error occurred when opening the file {exc}")
