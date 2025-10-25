import json
from loguru import logger as log


def txt_file_open(text: str) -> bytes:
    """
    Открывает и читает текстовый файл в бинарном режиме, возвращая его содержимое в виде байтов.

    :param text: Путь к файлу
    :return: Содержимое файла
    :raises FileNotFoundError: Если файл по указанному пути не найден.
    :raises Exception: Если произошла другая ошибка при открытии или чтении файла.
    """
    try:
        log.info(f"try to open text file: {text}")
        with open(text, "rb") as file:
            content = file.read()
            log.info(f"text file open successfully: {text}")
            return content
    except FileNotFoundError as not_found:
        log.error(f"Txt file not found(open): {not_found}")
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except Exception as e:
        log.error(f"Fail for open text file: {e}")
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
        log.info(f"try to save text file: {path}")
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        log.info(f"text file save successfully: {path}")
    except FileNotFoundError as not_found:
        log.error(f"Txt file not found (save): {not_found}")
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except Exception as e:
        log.error(f"Fail to save txt file: {e}")
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
        log.info(f"try to open JSON file: {path}")
        with open(path, mode="r", encoding="utf-8") as file:
            data = json.load(file)
            log.info(f"JSON opened successfully: {path}")
            return data
    except FileNotFoundError as not_found:
        log.error(f"JSON file not found: {not_found}")
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except json.JSONDecodeError as decode_error:
        log.error(f"JSON decode error in file {path}: {decode_error}")
        raise ValueError(f"Возникли проблемы с декодированием файла: {decode_error}")
    except Exception as e:
        log.error(f"Fail to open JSON file {path}: {e}")
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
        log.info(f"try to save JSON file: {path}")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=4)
        log.info(f"JSON file saved successfully: {path}")
    except Exception as e:
        log.error(f"Fail to save JSON file {path}: {e}")
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
        log.info(f"try to save binary file: {path}, size: {len(text)} byte")
        with open(path, "wb") as file:
            file.write(text)
        log.info(f"binary file save successfully: {path}")
    except FileNotFoundError as not_found:
        log.error(f"Bytes file not found: {not_found}")
        raise FileNotFoundError(f"Файл не найден: {not_found}")
    except Exception as e:
        log.error(f"Fail to save bytes file: {e}")
        raise Exception(f"Не удалось записать файл bytes {e}")