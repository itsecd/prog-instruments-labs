import json

import pandas as pd


def read_csv(filename: str) -> pd.DataFrame:
    """
    Читает CSV файл и возвращает DataFrame.

    :param filename: Имя CSV файла для чтения
    :return: DataFrame с данными из файла
    :raises Exception: Выводит сообщение об ошибке при проблемах чтения файла
    """
    try:
        with open(filename, mode="r", encoding="utf-16") as file:
            df = pd.read_csv(file, delimiter=";")
            return df
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
        return pd.DataFrame()


def read_json(filename: str) -> dict:
    """
    Читает JSON файл и возвращает словарь с данными.

    :param filename: Имя JSON файла для чтения
    :return: Словарь с данными из файла или пустой словарь при ошибке
    :raises FileNotFoundError: Если файл не найден
    :raises JSONDecodeError: Если ошибка формата JSON
    :raises Exception: Другие ошибки при чтении файла
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
    Записывает данные в JSON файл.

    :param filename: Имя файла для записи
    :param data: Словарь с данными для записи
    :raises PermissionError: Если нет прав для записи в файл
    :raises Exception: Другие ошибки при записи файла
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except PermissionError:
        print(f"No permission to write to file {filename}.")
    except Exception as exc:
        print(f"Error writing JSON: {exc}")
