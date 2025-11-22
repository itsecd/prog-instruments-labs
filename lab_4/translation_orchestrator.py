"""
Оркестратор процесса перевода.
"""

from translation_driver import TranslationDriver
from translation_service import TranslationService
from po_file_processor import POFileProcessor
from file_manager import FileManager
from logger import get_logger


class TranslationOrchestrator:
    """
    Управляет всем процессом перевода для одного языка.
    """

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.driver = None
        self.translation_service = None
        self.file_processor = None

    def __enter__(self):
        """Вход в контекстный менеджер."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера."""
        self.cleanup()

    def setup(self):
        """Настройка компонентов."""
        self.driver = TranslationDriver(self.config.driver_path, self.config.headless)
        self.translation_service = TranslationService(self.driver, self.config.max_retries)
        self.file_processor = POFileProcessor(self.translation_service)

    def cleanup(self):
        """Очистка ресурсов."""
        if self.driver:
            self.driver.close()

    def translate_language(self, language_code: str) -> bool:
        """
        Перевод PO файла для указанного языка.

        Args:
            language_code: Код языка

        Returns:
            bool: True если перевод успешен
        """
        try:
            # Навигация к переводчику
            self.driver.navigate_to_translator(
                self.config.interface_language,
                self.config.source_language,
                language_code
            )

            # Обработка PO файла
            file_path = FileManager.get_po_file_path(self.config.locale_path, language_code)
            return self.process_po_file(file_path)

        except Exception as e:
            self.logger.error(f"Ошибка перевода {language_code}: {e}")
            return False

    def process_po_file(self, file_path: str) -> bool:
        """
        Обработка и перевод PO файла.

        Args:
            file_path: Путь к PO файлу

        Returns:
            bool: True если обработка успешна
        """
        try:
            # Чтение файла
            content = FileManager.read_po_file(file_path)

            # Перевод содержимого
            translated_content = self.file_processor.process_file_content(content)

            # Сохранение если есть изменения
            if FileManager.file_has_changes(content, translated_content):
                FileManager.write_po_file(file_path, translated_content)
                self.logger.info(f"Файл сохранен: {file_path}")
                return True
            else:
                self.logger.info(f"Изменений нет: {file_path}")
                return True

        except FileNotFoundError:
            self.logger.error(f"Файл не найден: {file_path}")
            return False
        except Exception as e:
            self.logger.error(f"Ошибка обработки {file_path}: {e}")
            return False