import csv
import json

def read_json(path: str) -> dict:
    """
    Читает JSON файл и возвращает его содержимое в виде словаря.

    :param path: путь к JSON файлу для чтения
    :return: словарь с данными из JSON файла
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data: dict, path: str) -> None:
    """
    Записывает словарь в JSON файл.

    :param data: словарь с данными для записи
    :param path: путь к JSON файлу для записи
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_csv(path: str) -> list[dict]:
    """
    Читает CSV файл и возвращает его содержимое в виде списка словарей.

    :param path: путь к CSV файлу для чтения
    :return: список словарей, где каждый словарь представляет строку CSV файла
    """
    with open(path, 'r', encoding='utf-16', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        return list(reader)