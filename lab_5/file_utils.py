import json
import logging

logger = logging.getLogger(__name__)

def read(filename: str) -> str:
    """
    Читает содержимое текстового файла.
    :param filename: Путь к файлу, который нужно прочитать.
    :return: Строка с содержимым файла или сообщение об ошибке.
    """
    try:
        logger.info(f"Чтение файла: {filename}")
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
            logger.debug(f"Файл {filename} успешно прочитан, размер: {len(content)} символов")
            return content
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {filename}: {e}")

def save(filename: str, text: str) -> None:
    """
    Сохраняет текст в файл.
    :param filename: Путь к файлу, в который будет записан текст.
    :param text: Строка, которая будет записана в файл.
    """
    try:
        logger.info(f"Сохранение файла: {filename}")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        logger.debug(f"Файл {filename} успешно сохранен, размер: {len(text)} символов")
    except Exception as e:
        logger.error(f"Ошибка при записи в файл {filename}: {e}")