import json

import pandas as pd
from pandas import DataFrame


def read_csv(file_path: str, encoding: str = 'utf-16', delimiter: str = ';') -> DataFrame:
    """
    Чтение данных из CSV файла
    :param file_path: Путь к файлу
    :param encoding: Кодировка файла
    :param delimiter: Разделитель столбцов в CSV файле
    :return: DataFrame
    """
    try:
        return pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Файл пустой: {file_path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении CSV файла: {e}")


def read_json(file_path: str, encoding: str = 'utf-8') -> dict:
    """
    Чтение данных из JSON файла
    :param file_path: Путь к файлу
    :param encoding: Кодировка файла
    :return: Данные в виде словаря
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON файл не найден: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Ошибка декодирования JSON в файле {file_path}: {e}")
    except UnicodeDecodeError:
        raise ValueError(f"Ошибка кодировки в файле {file_path}.")
    except Exception as e:
        raise Exception(f"Ошибка при чтении JSON файла {file_path}: {e}")


def write_json(file_path: str, data: dict, encoding: str = 'utf-8') -> None:
    """
    Запись данных в JSON файл
    :param file_path: Путь к файлу для записи
    :param data: Данные для записи
    :param encoding: Кодировка файла
    :return: None
    """
    try:
        with open(file_path, 'w', encoding=encoding) as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as e:
        raise Exception(f"Ошибка записи JSON файла {file_path}: {e}")
    except FileNotFoundError as e:
        raise Exception(f"Путь не найден: {file_path}: {e}")