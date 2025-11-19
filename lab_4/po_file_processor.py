"""
PO file processing and translation management.
Handles reading, parsing, and translating PO file content.
"""

from typing import List, Tuple
from constants import POFileConstants, LogMessages
from po_parser import POParser
from translation_service import TranslationService


class POFileProcessor:
    """
    Processes PO files for translation, managing state during file processing.
    """

    def __init__(self, translation_service: TranslationService):
        """
        Initialize PO file processor.

        Args:
            translation_service: Service for handling text translation
        """
        self.translation_service = translation_service
        self._reset_state()

    def _reset_state(self):
        """Reset internal processing state."""
        self.current_translation: str = ""
        self.is_translating: bool = False
        self.is_complex_text: bool = False
        self.should_save_complex: bool = False
        self.complex_text_parts: List[str] = []

    def process_file_content(self, content: str) -> str:
        """
        Process and translate PO file content.

        Args:
            content: Original PO file content

        Returns:
            str: Translated PO file content
        """
        lines = content.splitlines(True)
        processed_lines = []

        for i, line in enumerate(lines):
            processed_line = self._process_line(line, lines, i)
            processed_lines.append(processed_line)

        return ''.join(processed_lines)

    def _process_line(self, current_line: str, all_lines: List[str], current_index: int) -> str:
        """
        Process a single line in the PO file.

        Args:
            current_line: Current line being processed
            all_lines: All lines in the PO file
            current_index: Index of current line

        Returns:
            str: Processed line (possibly with translation)
        """
        # Skip already translated strings
        if POParser.is_translated(all_lines, current_index, current_line):
            return current_line

        # Handle different line types
        if current_line.startswith(POFileConstants.MSGID_PREFIX):
            return self._handle_msgid_line(current_line, all_lines, current_index)
        elif self._is_complex_text_continuation(current_line):
            return self._handle_complex_text_line(current_line, all_lines, current_index)
        elif current_line.startswith(POFileConstants.MSGSTR_PREFIX) and self.is_translating:
            return self._handle_msgstr_line(current_line)
        else:
            return current_line

    def _handle_msgid_line(self, line: str, all_lines: List[str], index: int) -> str:
        """
        Handle msgid line - start of translatable string.

        Args:
            line: Current msgid line
            all_lines: All lines in PO file
            index: Current line index

        Returns:
            str: Original line (translation handled later)
        """
        text_content = POParser.extract_text_content(line)

        if text_content:  # Simple string
            self.current_translation = self.translation_service.translate(text_content)
            self.is_translating = True
        else:  # Complex multi-line string
            self.is_complex_text = True

        return line

    def _is_complex_text_continuation(self, line: str) -> bool:
        """
        Check if line is part of complex text continuation.

        Args:
            line: Line to check

        Returns:
            bool: True if line is complex text continuation
        """
        return (line.startswith(POFileConstants.QUOTE) and
                self.is_complex_text and
                len(line) > 2)

    def _handle_complex_text_line(self, line: str, all_lines: List[str], index: int) -> str:
        """
        Handle complex text continuation line.

        Args:
            line: Current complex text line
            all_lines: All lines in PO file
            index: Current line index

        Returns:
            str: Original line (translation handled later)
        """
        text_content = POParser.extract_text_content(line)
        self.complex_text_parts.append(text_content)

        # Check if next line starts msgstr (end of complex text)
        try:
            if all_lines[index + 1].startswith(POFileConstants.MSGSTR_PREFIX):
                self.is_complex_text = False
                self.should_save_complex = True
                self.is_translating = True
        except IndexError:
            print(LogMessages.SYNTAX_ERROR.format(line=index + 1, file="PO file"))

        return line

    def _handle_msgstr_line(self, line: str) -> str:
        """
        Handle msgstr line - write translation result.

        Args:
            line: Original msgstr line

        Returns:
            str: Translated msgstr line
        """
        if self.should_save_complex:
            result = self._write_complex_translation()
        else:
            result = self._write_simple_translation()

        self._reset_state()
        return result

    def _write_simple_translation(self) -> str:
        """
        Write translation for simple string.

        Returns:
            str: Formatted msgstr line with translation
        """
        return f'msgstr "{self.current_translation}"\n'

    def _write_complex_translation(self) -> str:
        """
        Write translation for complex multi-line string.

        Returns:
            str: Formatted msgstr lines with translation
        """
        full_text = ' '.join(self.complex_text_parts)
        translation = self.translation_service.translate(full_text)
        return f'msgstr ""\n"{translation}"\n'