import json

import pandas
import pandas as pd


def read_csv(csv_path: str) -> pandas.DataFrame:
    """
    Создает DataFrame из csv файла.
    :param csv_path путь к файлу csv в формате str
    :return:  DataFrame в формате pandas.DataFrame
    """
    try:
        with open(csv_path, 'r', encoding='utf-16') as f:
            df = pd.read_csv(f, delimiter=";")
            return df
    except FileNotFoundError:
        print(f"Файл не найден.")
    except Exception as e:
       print(f"Ошибка при загрузке данных: {e}")
    return pd.DataFrame()


def read_json(filename: str) -> dict:
    """
    Читает JSON.
    :param filename: Имя файла.
    :return: Содержимое JSON.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
    except json.JSONDecodeError:
        print(f"Файл {filename} не является корректным JSON.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    return {}