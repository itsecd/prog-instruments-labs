"""
PO file parsing utilities.
Handles detection of translated and untranslated strings.
"""

from constants import POFileConstants


class POParser:
    """
    Parser for PO files that can identify translated and untranslated strings.
    """

    @staticmethod
    def is_translated(lines: list[str], current_index: int, current_line: str) -> bool:
        """
        Check if a message is already translated in PO file.

        Args:
            lines: All lines from the PO file
            current_index: Index of the current line being processed
            current_line: The current line content

        Returns:
            bool: True if the message is already translated, False otherwise
        """
        if not current_line.startswith(POFileConstants.MSGID_PREFIX):
            return False

        # Simple one-line message
        if len(current_line) > POFileConstants.EMPTY_MSGID_LENGTH:
            return POParser._check_simple_translation(lines, current_index)
        # Multi-line message
        else:
            return POParser._check_complex_translation(lines, current_index)

    @staticmethod
    def _check_simple_translation(lines: list[str], index: int) -> bool:
        """
        Check translation status for simple one-line message.

        Args:
            lines: All lines from the PO file
            index: Current line index

        Returns:
            bool: True if simple message is translated
        """
        try:
            next_line = lines[index + 1]
            is_translated = (
                    next_line.startswith(POFileConstants.MSGSTR_PREFIX) and
                    len(next_line) > POFileConstants.EMPTY_MSGSTR_LENGTH
            )
            return is_translated
        except IndexError:
            return False

    @staticmethod
    def _check_complex_translation(lines: list[str], index: int) -> bool:
        """
        Check translation status for multi-line message.

        Args:
            lines: All lines from the PO file
            index: Current line index

        Returns:
            bool: True if complex message is translated
        """
        try:
            line_offset = 1
            while True:
                current_line = lines[index + line_offset]
                line_offset += 1

                if current_line.startswith(POFileConstants.MSGSTR_PREFIX):
                    next_line = lines[index + line_offset]
                    has_translation = (
                            len(current_line) > POFileConstants.EMPTY_MSGSTR_LENGTH or
                            (next_line.startswith(POFileConstants.QUOTE) and len(next_line) > 2)
                    )
                    return has_translation
        except IndexError:
            return False

    @staticmethod
    def extract_text_content(line: str) -> str:
        """
        Extract text content from a PO file line.

        Args:
            line: Line from PO file

        Returns:
            str: Extracted text content
        """
        start_pos = line.find(POFileConstants.QUOTE) + 1
        end_pos = line.rfind(POFileConstants.QUOTE)
        return line[start_pos:end_pos].strip()