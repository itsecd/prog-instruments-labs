import os
import pytest
from unittest.mock import mock_open, patch

from assymetric import RSA
from file_operation import read_json, write_file, read_file, write_bytes_to_file, read_bytes_from_file
from symmetric import CAST_5


def test_rsa_key_generation():
    rsa = RSA()
    rsa.generate_key(2048)
    assert rsa.public_key is not None
    assert rsa.private_key is not None


@pytest.mark.parametrize("message", [
    b'Test message',
    b'Test message with numbers 12345',
    b'Special characters: !@#$%^&*()',
])
def test_rsa_encryption_decryption_parametrized(message):
    rsa = RSA()
    rsa.generate_key(2048)
    encrypted_message = rsa.encrypt_bytes(message)
    decrypted_message = rsa.decrypt_bytes(encrypted_message)
    assert decrypted_message == message


@pytest.mark.parametrize("message", [
    b'Test message',
    b'Test message with numbers 12345',
    b'Special characters: !@#$%^&*()',
])
def test_encrypt_decrypt_roundtrip(message):
    cast5 = CAST_5()
    cast5.generate_key()

    ciphertext = cast5.encrypt_bytes(message)
    decrypted = cast5.decrypt_bytes(ciphertext)

    assert decrypted == message


def test_cast5_key_generation():
    cast5 = CAST_5()
    cast5.generate_key(16)
    assert cast5.key is not None
    assert len(cast5.key) == 16


@pytest.mark.parametrize("json_data, expected_output", [
    ('{"key": "value"}', {"key": "value"}),
    ('{"number": 123}', {"number": 123}),
    ('{"boolean": true}', {"boolean": True}),
    ('{"list": [1, 2, 3]}', {"list": [1, 2, 3]})
])
def test_read_json_parametrized(json_data, expected_output):
    with patch("builtins.open", mock_open(read_data=json_data)):
        result = read_json("dummy_path.json")
        assert result == expected_output


def test_cast5_encrypt_empty_data():
    cast5 = CAST_5()
    cast5.generate_key(16)
    encrypted = cast5.encrypt_bytes(b'')
    assert len(encrypted) > 0


def test_rsa_encrypt_invalid_data():
    rsa = RSA()
    rsa.generate_key(2048)
    try:
        rsa.encrypt_bytes(None)
        assert False
    except TypeError:
        assert True


def test_cast5_ciphertext_format():
    cast5 = CAST_5()
    cast5.generate_key(16)
    plaintext = b'Check message for CAST5'
    ciphertext = cast5.encrypt_bytes(plaintext)
    assert isinstance(ciphertext, bytes)


def test_incompatibility_between_algorithms():
    cast5 = CAST_5()
    cast5.generate_key(16)
    rsa = RSA()
    rsa.generate_key(2048)
    original_message = b'Test message'

    encrypted_with_cast5 = cast5.encrypt_bytes(original_message)
    try:
        rsa.decrypt_bytes(encrypted_with_cast5)
        assert False
    except Exception:
        assert True


def test_write_file_success():
    temp_file = "test_file.txt"
    test_data = "This is a test message."

    write_file(temp_file, test_data)

    with open(temp_file, "r", encoding="utf-8") as file:
        written_data = file.read()

    assert written_data == test_data

    os.remove(temp_file)


def test_read_file_success():
    temp_file = "test_file.txt"
    test_data = "This is a test message."

    with open(temp_file, "w", encoding="utf-8") as file:
        file.write(test_data)

    read_data = read_file(temp_file)

    assert read_data == test_data

    os.remove(temp_file)


def test_write_bytes_to_file_success():
    temp_file = "test_file.bin"
    test_data = b"This is a test byte message."

    write_bytes_to_file(temp_file, test_data)

    with open(temp_file, "rb") as file:
        written_data = file.read()

    assert written_data == test_data

    os.remove(temp_file)


def test_read_bytes_from_file_success():
    temp_file = "test_file.bin"
    test_data = b"This is a test byte message."

    with open(temp_file, "wb") as file:
        file.write(test_data)

    read_data = read_bytes_from_file(temp_file)

    assert read_data == test_data

    os.remove(temp_file)
