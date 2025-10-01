import csv
import json

import csv

def load_csv(path):
    """
    Загружает CSV и возвращает список словарей
    :param path: путь к csv файлу
    :return: list[dict]
    """
    try:
        with open(path, "r", encoding="utf-16", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            return list(reader)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"[Ошибка] Файл '{path}' не найден.")

    except UnicodeDecodeError:
        raise UnicodeDecodeError("utf-16", b"", 0, 1, f"[Ошибка] Не удалось декодировать файл '{path}")

    except csv.Error as e:
        raise csv.Error(f"[Ошибка] Проблема с CSV файлом '{path}': {e}")

    except Exception as e:
        raise Exception(f"[Ошибка] Непредвиденная ошибка при работе с '{path}': {e}")


def load_json(path):
    """
    Loads a json file
    :param path: path to json file
    :return: json object
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"[Ошибка] JSON файл '{path}' не найден.")

    except Exception as e:
        raise Exception(f"[Ошибка] Непредвиденная ошибка при работе с '{path}': {e}")
