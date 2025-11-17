import json
import logging

logger = logging.getLogger(__name__)

def read_binary_file(file_path: str) -> bytes:
    """
    Читает бинарный файл.
    :param file_path: Путь к файлу
    :return: Содержимое файла или пустой bytes при ошибке
    """
    try:
        logger.debug(f"Чтение бинарного файла: {file_path}")
        with open(file_path, 'rb') as f:
            data = f.read()
            logger.debug(f"Бинарный файл {file_path} прочитан, размер: {len(data)} байт")
            return data
    except Exception as e:
        logger.error(f"Ошибка чтения бинарного файла {file_path}: {e}")
        raise

def write_binary_file(file_path: str, data: bytes):
    """
    Записывает данные в бинарный файл.
    :param file_path: Путь к файлу
    :param data: Данные для записи
    """
    try:
        logger.debug(f"Запись бинарного файла: {file_path}, размер: {len(data)} байт")
        with open(file_path, 'wb') as f:
            f.write(data)
        logger.debug(f"Бинарный файл {file_path} успешно записан")
    except Exception as e:
        logger.error(f"Ошибка записи в бинарный файл {file_path}: {e}")

def load_json_config(config_path: str) -> dict:
    """
    Загружает конфигурацию из JSON файла.
    :param config_path: Путь к файлу
    :return: Словарь с конфигурацией или пустой dict при ошибке
    """
    try:
        logger.debug(f"Загрузка конфигурации из {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.debug(f"Конфигурация из {config_path} успешно загружена")
            return config
    except Exception as e:
        logger.error(f"Ошибка чтения файла {config_path}: {e}")
        return {}