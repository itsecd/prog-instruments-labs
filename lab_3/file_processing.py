import csv
import json


def read_json(filename: str) -> dict:
    """
    Читает данные из json-файла и возвращает их в виде словаря.
    :param filename: путь к json-файлу
    :return: словарь с данными из json-файла
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found")
    except json.JSONDecodeError:
        print(f"File {filename} isn't correct JSON")
    except Exception as e:
        print(f"An error occurred while reading the file {filename}: {e}")


def write_json(data: dict, filename: str) -> None:
    """
    Записывает данные в json-файл.
    :param data: словарь с данными для записи
    :param filename: путь к json-файлу, в который будут записаны данные
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"An error occurred while saving the file {filename}: {e}")


def read_csv(filename: str) -> list[dict]:
    """
    Читает данные из csv-файла и возвращает их в виде списка словарей.
    :param filename: путь к csv-файлу
    :return: список словарей с данными из строк csv-файла
    """
    try:
        with open(filename, "r", encoding="utf-16", newline="") as file:
            reader = csv.DictReader(file, delimiter=";")
            return list(reader)
    except FileNotFoundError:
        print(f"File {filename} not found")
    except Exception as e:
        print(f"An error occurred while reading the file {filename}: {e}")
