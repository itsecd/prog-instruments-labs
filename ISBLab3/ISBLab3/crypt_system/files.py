import json

def txt_file_open(text: str) -> bytes:
    """
    Открывает и читает текстовый файл в бинарном режиме, возвращая его содержимое в виде байтов.

    :param text: Путь к файлу
    :return: Содержимое файла
    :raises FileNotFoundError: Если файл по указанному пути не найден.
    :raises Exception: Если произошла другая ошибка при открытии или чтении файла.
    """
    try:
        with open(text, "rb") as file:
            return file.read()
    except FileNotFoundError as not_found:
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except Exception as e:
        raise Exception(f"Не удалось открыть текстовый файл {e}")


def txt_file_save(text: str, path: str) -> None:
    """
    Сохраняет текстовые данные в файл

    :param text: Текст который нужно записать в файл.
    :param path: Путь к файлу
    :return: None
    :raises FileNotFoundError: Если не удалось найти или создать файл по указанному пути.
    :raises Exception: Если произошла другая ошибка при записи в файл.
    """
    try:
        with open(path, "w") as file:
            file.write(text)
    except FileNotFoundError as not_found:
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except Exception as e:
        raise Exception(f"Не удалось сохранить текстовый файл {e}")


def json_file_open(path: str) -> dict:
    """
    Открывает JSON-файл и возвращает его содержимое в виде словаря.

    :param path: Путь к JSON-файлу
    :return: Словарь, содержащий данные из JSON-файла.
    :raises FileNotFoundError: Если файл по указанному пути не найден.
    :raises ValueError: Если файл содержит некорректный JSON (ошибка декодирования).
    :raises Exception: Если произошла другая ошибка при открытии или чтении файла.
    """
    try:
        with open(path, mode="r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as not_found:
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except json.JSONDecodeError as decode_error:
        raise ValueError(f"Возникли проблемы с декодированием файла: {decode_error}")
    except Exception as e:
        raise Exception(f"Не удалось открыть файл {e}")


def json_file_save(path: str, info) -> None:
    """
    Сохраняет данные в JSON-файл

    :param path: Путь к файлу
    :param info: Данные, которые нужно сериализовать в JSON.
    :return: None
    :raises Exception: Если произошла ошибка при записи в файл
    """
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Не удалось сохранить файл json: {e}")


def bytes_file_save(path: str, text: bytes) -> None:
    """
    Сохраняет бинарные данные в файл.

    :param path: Путь к файлу
    :param text: Бинарные данные, которые нужно записать.
    :return: None
    :raises FileNotFoundError: Если не удалось найти или создать файл по указанному пути.
    :raises Exception: Если произошла другая ошибка при записи в файл.
    """
    try:
        with open(path, "wb") as file:
            file.write(text)
    except FileNotFoundError as not_found:
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except Exception as e:
        raise Exception(f"Не удалось записать файл bytes {e}")
