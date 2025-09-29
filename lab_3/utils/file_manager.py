import csv
import json
from pandas import DataFrame
import pandas as pd


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


def read_json(file_path: str, encoding: str = 'utf-16') -> dict:
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
