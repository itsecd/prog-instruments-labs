"""
High-level translation service that coordinates text processing and translation.
"""

from typing import Optional
from constants import LogMessages
from translation_driver import TranslationDriver
from text_processor import TextProcessor


class TranslationService:
    """
    High-level service that coordinates text preprocessing, translation, and postprocessing.
    """

    def __init__(self, driver: TranslationDriver, max_retries: int = 3):
        """
        Initialize translation service.

        Args:
            driver: TranslationDriver instance for browser automation
            max_retries: Maximum number of translation retry attempts
        """
        self.driver = driver
        self.max_retries = max_retries
        self.last_translation: Optional[str] = None

    def translate(self, text: str) -> str:
        """
        Translate text with full preprocessing and postprocessing.

        Args:
            text: Text to translate

        Returns:
            str: Translated text or empty string if translation failed
        """
        if not text.strip():
            return text

        # Preprocess text for translation
        processed_text, variables, html_classes, has_service_chars, has_unnamed_format = (
            TextProcessor.preprocess_text(text)
        )

        # Perform translation
        raw_translation = self._execute_translation(processed_text)

        if not raw_translation:
            return ""

        # Postprocess translation to restore original patterns
        final_translation = TextProcessor.postprocess_text(
            raw_translation, variables, html_classes, has_service_chars, has_unnamed_format
        )

        self._log_translation_success(text, final_translation)
        return final_translation

    def _execute_translation(self, text: str) -> str:
        """
        Execute the actual translation using the driver.

        Args:
            text: Preprocessed text to translate

        Returns:
            str: Raw translation result
        """
        return self.driver.translate_text(text, self.max_retries)

    def _log_translation_success(self, source_text: str, translated_text: str):
        """
        Log successful translation.

        Args:
            source_text: Original source text
            translated_text: Successfully translated text
        """
        print(LogMessages.TRANSLATION_SUCCESS.format(
            source=source_text,
            translation=translated_text
        ))

    def update_settings(self, max_retries: Optional[int] = None):
        """
        Update translation service settings.

        Args:
            max_retries: New maximum retry attempts (if provided)
        """
        if max_retries is not None:
            self.max_retries = max_retries