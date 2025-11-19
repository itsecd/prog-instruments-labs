"""
Конфигурация приложения для перевода PO файлов.
"""

from dataclasses import dataclass
from typing import List
import os


@dataclass
class TranslationConfig:
    """Основная конфигурация для перевода."""
    driver_path: str
    locale_path: str
    headless: bool = True
    interface_language: str = 'en'
    source_language: str = 'en'
    max_retries: int = 3
    multi_process: bool = False
    max_processes: int = 10

    def validate(self):
        """Проверка корректности конфигурации."""
        if not os.path.exists(self.driver_path):
            raise ValueError(f"Путь к драйверу не существует: {self.driver_path}")

        if not os.path.exists(self.locale_path):
            raise ValueError(f"Путь к локалям не существует: {self.locale_path}")


@dataclass
class BatchConfig:
    """Конфигурация для пакетной обработки."""
    language_codes: List[str]
    translation_config: TranslationConfig

    def validate(self):
        """Проверка корректности пакетной конфигурации."""
        if not self.language_codes:
            raise ValueError("Список языков не может быть пустым")

        self.translation_config.validate()