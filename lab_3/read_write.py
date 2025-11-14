import pandas
import pandas as pd
import json


def read_csv(csv_path: str) -> pandas.DataFrame:
    """
    Создает DataFrame из csv файла.
    :param csv_path путь к файлу csv в формате str
    :return:  DataFrame в формате pandas.DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise Exception(f"Ошибка при загрузке данных: {e}")

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