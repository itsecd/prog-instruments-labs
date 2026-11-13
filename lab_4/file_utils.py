import json
from pathlib import Path


def read_json_file(file_path: str) -> dict:
    """
    Читает JSON-файл.

    :param file_path: путь к JSON-файлу
    :return: данные из JSON-файла
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Файл не найден: {file_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Ошибка чтения JSON-файла: {file_path}") from error


def write_json_file(file_path: str, data: dict) -> None:
    """
    Записывает словарь в JSON-файл.

    :param file_path: путь к JSON-файлу
    :param data: данные для записи
    :return: None
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except OSError as error:
        raise OSError(f"Ошибка записи JSON-файла: {file_path}") from error


def read_binary_file(file_path: str) -> bytes:
    """
    Читает бинарный файл.

    :param file_path: путь к файлу
    :return: данные из файла
    """
    try:
        with open(file_path, "rb") as file:
            return file.read()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Файл не найден: {file_path}") from error


def write_binary_file(file_path: str, data: bytes) -> None:
    """
    Записывает данные в бинарный файл.

    :param file_path: путь к файлу
    :param data: данные для записи
    :return: None
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as file:
            file.write(data)
    except OSError as error:
        raise OSError(f"Ошибка записи файла: {file_path}") from error
