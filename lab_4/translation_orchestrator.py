"""
Orchestrates the complete translation workflow.
Coordinates all components for seamless PO file translation.
"""

from typing import Optional, List
from constants import LogMessages
from translation_driver import TranslationDriver
from translation_service import TranslationService
from po_file_processor import POFileProcessor
from file_manager import FileManager
from config import LanguageConfig, TranslationConfig


class TranslationOrchestrator:
    """
    Orchestrates the complete translation workflow for a language.
    Manages the entire pipeline from file reading to translation and saving.
    """

    def __init__(self, config: TranslationConfig):
        """
        Initialize translation orchestrator.

        Args:
            config: Translation configuration
        """
        self.config = config
        self.driver: Optional[TranslationDriver] = None
        self.translation_service: Optional[TranslationService] = None
        self.file_processor: Optional[POFileProcessor] = None

    def setup(self):
        """Setup all components for translation."""
        self.driver = TranslationDriver(
            self.config.driver_path,
            self.config.headless
        )

        self.translation_service = TranslationService(
            self.driver,
            self.config.max_retries
        )

        self.file_processor = POFileProcessor(self.translation_service)

    def translate_language(self, language_code: str) -> bool:
        """
        Translate PO file for a specific language.

        Args:
            language_code: Language code to translate

        Returns:
            bool: True if translation was successful, False otherwise
        """
        language_config = LanguageConfig(language_code, self.config)

        try:
            language_config.validate()
            return self._execute_translation(language_config)

        except Exception as error:
            print(f"[!] Failed to translate {language_code}: {error}")
            return False

    def _execute_translation(self, language_config: LanguageConfig) -> bool:
        """
        Execute the translation pipeline for a language.

        Args:
            language_config: Language-specific configuration

        Returns:
            bool: True if translation completed successfully
        """
        if not self.driver or not self.translation_service or not self.file_processor:
            self.setup()

        try:
            # Navigate to translator
            self.driver.navigate_to_translator(
                self.config.interface_language,
                self.config.source_language,
                language_config.language_code
            )

            # Process PO file
            return self._process_po_file(language_config.po_file_path)

        except Exception as error:
            print(f"[!] Translation error for {language_config.language_code}: {error}")
            return False

    def _process_po_file(self, po_file_path: str) -> bool:
        """
        Process and translate a PO file.

        Args:
            po_file_path: Path to PO file

        Returns:
            bool: True if processing was successful
        """
        try:
            # Read original content
            original_content = FileManager.read_po_file(po_file_path)

            # Process and translate content
            translated_content = self.file_processor.process_file_content(original_content)

            # Save if changes were made
            if FileManager.file_has_changes(original_content, translated_content):
                FileManager.write_po_file(po_file_path, translated_content)
                return True
            else:
                FileManager.log_no_changes(po_file_path)
                return True  # Still successful, just no changes

        except FileNotFoundError:
            return False
        except Exception as error:
            print(f"[!] PO file processing error for {po_file_path}: {error}")
            return False

    def cleanup(self):
        """Cleanup resources."""
        if self.driver:
            self.driver.close()
            self.driver = None

        self.translation_service = None
        self.file_processor = None

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()