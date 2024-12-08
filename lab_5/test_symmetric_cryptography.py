"""
Unit tests for the SymmetricCryptography class.
"""

import pytest
from SymmetricCryptography import SymmetricCryptography


class TestSymmetricCryptography:
    @pytest.fixture
    def symmetric_crypto(self) -> SymmetricCryptography:
        """
        Fixture to provide an instance of SymmetricCryptography.
        """
        return SymmetricCryptography(key_len=256)

    def test_generate_key(self, symmetric_crypto: SymmetricCryptography) -> None:
        """
        Test the generation of a symmetric key.
        """
        key = symmetric_crypto.generate_key()
        assert len(key) == 32  # Ensure the key length matches the specified key length

    def test_encrypt_decrypt_text(self, symmetric_crypto: SymmetricCryptography) -> None:
        """
        Test encryption and decryption of text using a symmetric key.
        """
        symmetric_key = symmetric_crypto.generate_key()
        text = b"Test data for encryption"
        encrypted_text = symmetric_crypto.encrypt_text(symmetric_key, text)
        assert encrypted_text != text  # Ensure the text is encrypted

        decrypted_text = symmetric_crypto.decrypt_text(symmetric_key, encrypted_text)
        assert decrypted_text == text  # Ensure the decrypted text matches the original
