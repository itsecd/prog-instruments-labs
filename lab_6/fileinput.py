import json


def file_open(original_text):
    """
    Funtion open files
    :param original_text:
    :return:
    """
    try:
        with open(original_text, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Файл не найден: {original_text}")
    except Exception as e:
        print(f"Не удалось открыть файл: {e}")


def file_save(path, text):
    """
    Funtion save files
    :param path: path to save
    :param text: encoded text
    :return: None
    """
    try:
        with open(path, "w", encoding="utf8") as file:
            file.write(text)
    except Exception as e:
        print(f"Не удалось записать файл: {e}")


def json_file_open(json_file):
    """
    Function open json files
    :param json_file: json file
    :return: text from file
    """
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Файл не найден: {json_file}")


def json_file_save(path, info):
    """
    Function save info to JSON files
    :param path: path to save file
    :param info: information
    :return: None
    """
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Не удалось записать файл: {e}")
