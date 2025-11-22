"""
Text preprocessing and postprocessing utilities.
Handles special patterns and formatting in translatable text.
"""

from constants import POFileConstants, TranslatorConstants
from typing import Tuple, List


class TextProcessor:
    """
    Handles text preprocessing and postprocessing for translation.
    Manages Python formatting, HTML classes, and escape sequences.
    """

    @staticmethod
    def preprocess_text(text: str) -> Tuple[str, List[str], List[str], bool, bool]:
        """
        Prepare text for translation by handling special patterns.

        Args:
            text: Original text to preprocess

        Returns:
            Tuple containing:
            - Processed text
            - List of Python format variables
            - List of HTML classes
            - Boolean indicating if service characters were found
            - Boolean indicating if unnamed formatting exists
        """
        variables = []
        html_classes = []
        has_service_chars = False
        has_unnamed_format = POFileConstants.FORMAT_PYTHON_UNNAMED in text

        # Handle Python named formatting
        text = TextProcessor._handle_python_named_formatting(text, variables)

        # Handle escape sequences
        text, has_service_chars = TextProcessor._handle_escape_sequences(text)

        # Handle HTML classes
        text = TextProcessor._handle_html_classes(text, html_classes)

        return text, variables, html_classes, has_service_chars, has_unnamed_format

    @staticmethod
    def postprocess_text(
            text: str,
            variables: List[str],
            html_classes: List[str],
            has_service_chars: bool,
            has_unnamed_format: bool
    ) -> str:
        """
        Restore original patterns in translated text.

        Args:
            text: Translated text to postprocess
            variables: Python format variables to restore
            html_classes: HTML classes to restore
            has_service_chars: Whether to restore service characters
            has_unnamed_format: Whether to handle unnamed formatting

        Returns:
            str: Fully processed text with original patterns restored
        """
        if has_unnamed_format:
            text = text.replace('%S', '%s')

        text = TextProcessor._restore_python_named_formatting(text, variables)
        text = TextProcessor._restore_html_classes(text, html_classes)
        text = TextProcessor._restore_escape_sequences(text, has_service_chars)

        return text

    @staticmethod
    def _handle_python_named_formatting(text: str, variables: List[str]) -> str:
        """Replace Python named formatting with temporary placeholders."""
        format_count = text.count(POFileConstants.FORMAT_PYTHON_NAMED)

        for i in range(format_count):
            format_start = text.find(POFileConstants.FORMAT_PYTHON_NAMED)
            format_end = text[format_start:].find(')s') + 2  # +2 to include ')s'

            if format_end > 2:  # Valid format found
                named_format = text[format_start:format_start + format_end]
                variables.append(named_format)
                substitution = create_substitution_pattern(i + 1)
                text = text.replace(named_format, substitution, 1)

        return text

    @staticmethod
    def _restore_python_named_formatting(text: str, variables: List[str]) -> str:
        """Restore Python named formatting from temporary placeholders."""
        for i, variable in enumerate(variables):
            substitution = create_substitution_pattern(i + 1)
            text = text.replace(substitution, variable, 1)
        return text

    @staticmethod
    def _handle_escape_sequences(text: str) -> Tuple[str, bool]:
        """Handle escape sequences in text."""
        has_service_chars = POFileConstants.ESCAPED_QUOTE in text
        if has_service_chars:
            text = text.replace(POFileConstants.ESCAPED_QUOTE, POFileConstants.QUOTE)
        return text, has_service_chars

    @staticmethod
    def _restore_escape_sequences(text: str, has_service_chars: bool) -> str:
        """Restore escape sequences in text."""
        if has_service_chars:
            text = text.replace(POFileConstants.QUOTE, POFileConstants.ESCAPED_QUOTE)
        return text

    @staticmethod
    def _handle_html_classes(text: str, html_classes: List[str]) -> str:
        """Replace HTML classes with temporary placeholders."""
        class_count = text.count(POFileConstants.HTML_CLASS_PREFIX)

        for i in range(class_count):
            class_start = text.find(POFileConstants.HTML_CLASS_PREFIX)
            class_end = text[class_start + 7:].find(
                POFileConstants.QUOTE) + 7 + 1  # +7 for 'class="', +1 for closing quote

            if class_end > 8:  # Valid class found
                html_class = text[class_start:class_start + class_end]
                html_classes.append(html_class)
                text = text.replace(html_class, TranslatorConstants.TEMP_SUBSTITUTION, 1)

        return text

    @staticmethod
    def _restore_html_classes(text: str, html_classes: List[str]) -> str:
        """Restore HTML classes from temporary placeholders."""
        for html_class in html_classes:
            text = text.replace(TranslatorConstants.TEMP_SUBSTITUTION, html_class, 1)
        return text


def create_substitution_pattern(index: int) -> str:
    """Create substitution pattern for temporary replacements."""
    return TranslatorConstants.SUBSTITUTION_PATTERN % (index,)