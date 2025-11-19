"""
File management utilities for PO file operations.
Handles reading, writing, and file path management.
"""

import os
from constants import LogMessages


class FileManager:
    """
    Manages file operations for PO file translation.
    """

    @staticmethod
    def read_po_file(file_path: str) -> str:
        """
        Read PO file content.

        Args:
            file_path: Path to PO file

        Returns:
            str: File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            with open(file_path, 'r', encoding='UTF-8') as file:
                content = file.read()
                print(LogMessages.FILE_OPENED.format(path=file_path))
                return content
        except FileNotFoundError:
            print(LogMessages.FILE_NOT_FOUND.format(path=file_path))
            raise

    @staticmethod
    def write_po_file(file_path: str, content: str):
        """
        Write content to PO file.

        Args:
            file_path: Path to PO file
            content: Content to write
        """
        with open(file_path, 'w', encoding='UTF-8') as file:
            file.write(content)
        print(LogMessages.FILE_SAVED.format(path=file_path))

    @staticmethod
    def get_po_file_path(locale_path: str, language_code: str) -> str:
        """
        Get full path to PO file for specific language.

        Args:
            locale_path: Base locale directory path
            language_code: Language code (e.g., 'de', 'fr')

        Returns:
            str: Full path to PO file
        """
        return os.path.join(locale_path, language_code, 'LC_MESSAGES', 'django.po')

    @staticmethod
    def file_has_changes(original_content: str, new_content: str) -> bool:
        """
        Check if file content has changed.

        Args:
            original_content: Original file content
            new_content: New file content

        Returns:
            bool: True if content has changed
        """
        return original_content != new_content

    @staticmethod
    def log_no_changes(file_path: str):
        """Log message when no changes are made to file."""
        print(LogMessages.NO_CHANGES.format(path=file_path))