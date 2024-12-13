import pytest
from unittest.mock import mock_open, patch
from cryptography.hazmat.primitives.asymmetric import rsa
from asymmetric import Asymmetric
from symmetric import Symmetric
from file_readers import read_json_file, write_key, write_file, read_txt
import os

# Test for Symmetric encryption
class TestSymmetric:

    def test_generate_key(self):
        sym = Symmetric()
        key = sym.generate_key()
        assert len(key) == 16

    def test_encrypt_decrypt_text(self):
        sym = Symmetric()
        sym.generate_key()
        original_text = b"Test-symmetric-encryption"
        encrypted = sym.encrypt_text(original_text)
        decrypted = sym.decrypt_text(encrypted)
        assert decrypted == original_text.decode('utf-8')

# Test for Asymmetric encryption
class TestAsymmetric:

    def test_generate_key(self):
        asym = Asymmetric()
        asym.generate_key()
        assert isinstance(asym.private_key, rsa.RSAPrivateKey)
        assert isinstance(asym.public_key, rsa.RSAPublicKey)

    def test_encrypt_decrypt_key(self):
        asym = Asymmetric()
        asym.generate_key()
        symmetric_key = os.urandom(16)
        encrypted_key = asym.encrypt_key(symmetric_key)
        decrypted_key = asym.decrypt_key(encrypted_key)
        assert symmetric_key == decrypted_key

# Test for file operations
class TestFileReaders:

    def test_read_json_file(self):
        mock_data = '{"key": "value"}'
        with patch("builtins.open", mock_open(read_data=mock_data)) as mock_file:
            result = read_json_file("dummy_path.json")
            assert result == {"key": "value"}

    def test_write_key(self):
        mock_key = b"123456"
        with patch("builtins.open", mock_open()) as mock_file:
            write_key("dummy_path.key", mock_key)
            mock_file.assert_called_once_with("dummy_path.key", "wb")

    @pytest.mark.parametrize("data, expected", [(b"text data", b"text data"), ("text data", "text data")])
    def test_write_file(self, data, expected):
        with patch("builtins.open", mock_open()) as mock_file:
            write_file("dummy_path.txt", data)
            mock_file().write.assert_called_once_with(expected)

    def test_read_txt(self):
        mock_data = b"Some text data"
        with patch("builtins.open", mock_open(read_data=mock_data)) as mock_file:
            result = read_txt("dummy_path.txt")
            assert result == mock_data

# Advanced testing: parametrization and mocks
class TestIntegration:

    @pytest.mark.parametrize("text", [b"Hello-World", b"Another-Test-Text"])
    def test_encrypt_decrypt_integration(self, text):
        sym = Symmetric()
        sym.generate_key()
        encrypted = sym.encrypt_text(text)
        decrypted = sym.decrypt_text(encrypted)
        assert decrypted == text.decode('utf-8')

    def test_asymmetric_serialization(self, tmp_path):
        private_path = tmp_path / "private.pem"
        public_path = tmp_path / "public.pem"
        asym = Asymmetric()
        asym.generate_key()
        asym.serialize_private_key(str(private_path))
        asym.serialize_public_key(str(public_path))

        new_asym = Asymmetric()
        new_asym.deserialize_private_key(str(private_path))
        new_asym.deserialize_public_key(str(public_path))

        assert isinstance(new_asym.private_key, rsa.RSAPrivateKey)
        assert isinstance(new_asym.public_key, rsa.RSAPublicKey)
