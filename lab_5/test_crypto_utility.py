"""
Unit tests for utility functions used in cryptographic operations.
"""

import pytest
from CryptoUtility import CryptoUtility
from pathlib import Path

class TestCryptoUtility:
    def test_serialize_and_deserialize_key(self, tmp_path: Path) -> None:
        """
        Test the serialization and deserialization of a cryptographic key.
        """
        key = b"SampleKeyData123456"
        key_path = tmp_path / "key.txt"

        CryptoUtility.serialize_key(key, str(key_path))
        deserialized_key = CryptoUtility.deserialize_key(str(key_path))

        assert deserialized_key == key  # Ensure key matches after deserialization

    def test_read_and_write_text_file(self, tmp_path: Path) -> None:
        """
        Test writing and reading of a text file.
        """
        text_path = tmp_path / "text.txt"
        content = "Test content for text file"

        CryptoUtility.write_text_file(content.encode(), str(text_path))
        read_content = CryptoUtility.read_text_file(str(text_path), "rb")

        assert read_content.decode() == content  # Ensure written and read content match
