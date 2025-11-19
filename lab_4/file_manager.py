"""
Управление файлами для PO файлов.
"""

import os
from logger import get_logger


class FileManager:
    """Менеджер файловых операций."""

    @staticmethod
    def read_po_file(file_path: str) -> str:
        """
        Чтение PO файла.

        Args:
            file_path: Путь к файлу

        Returns:
            str: Содержимое файла
        """
        logger = get_logger()
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.info(f"Файл прочитан: {file_path}")
                return content
        except FileNotFoundError:
            logger.error(f"Файл не найден: {file_path}")
            raise

    @staticmethod
    def write_po_file(file_path: str, content: str):
        """
        Запись PO файла.

        Args:
            file_path: Путь к файлу
            content: Содержимое для записи
        """
        logger = get_logger()
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info(f"Файл записан: {file_path}")

    @staticmethod
    def get_po_file_path(locale_path: str, language_code: str) -> str:
        """
        Получение пути к PO файлу.

        Args:
            locale_path: Путь к локалям
            language_code: Код языка

        Returns:
            str: Полный путь к файлу
        """
        return os.path.join(locale_path, language_code, 'LC_MESSAGES', 'django.po')

    @staticmethod
    def file_has_changes(original: str, new: str) -> bool:
        """
        Проверка наличия изменений.

        Args:
            original: Исходное содержимое
            new: Новое содержимое

        Returns:
            bool: True если есть изменения
        """
        return original != new