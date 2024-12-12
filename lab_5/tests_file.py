import pytest
import os
import random
from cryptography.hazmat.primitives.asymmetric import rsa
from unittest.mock import patch

from files_funct import FilesFunct
from algoritms.tripldes import SymmetricKey
from algoritms.rsa import AsymmetricKey


@pytest.fixture
def generate_sim_key():
    key_len = random.choice([64, 128, 192])
    key = os.urandom(key_len // 8)
    return key


@pytest.fixture
def symmetric():
    return SymmetricKey()


@pytest.fixture
def asymmetric():
    return AsymmetricKey()


@pytest.mark.parametrize("incorrect_len", [0, 8, 16])
def test_incorrect_len_key(incorrect_len):
    with patch('builtins.input', side_effect=[str(incorrect_len), '128']):
        symmetric_key = SymmetricKey()
        key_len = symmetric_key.selecting_key_len()
        assert key_len == 128


@pytest.mark.parametrize("correct_len", [64, 128, 192])
def test_correct_generate_key(correct_len):
    symmetric_key = SymmetricKey()
    key = symmetric_key.generate_key(correct_len)
    assert len(key) * 8 == correct_len


def test_key_generation(asymmetric):
    public_key, private_key = asymmetric.generate_keys()
    assert isinstance(public_key, rsa.RSAPublicKey)
    assert isinstance(private_key, rsa.RSAPrivateKey)


def test_generate_key(symmetric):
    key_length = 128
    generated_key = symmetric.generate_key(key_length)
    assert len(generated_key) == key_length // 8


def test_bytes_from_files(generate_sim_key):
    file_name = "test_key.txt"
    FilesFunct.write_bytes_to_file(file_name, generate_sim_key)
    new_key = FilesFunct.read_bytes_from_file(file_name)
    assert new_key == generate_sim_key
    os.remove("test_key.txt")


def test_text_from_file():
    file_name = "test_text.txt"
    text = "test text in file"
    FilesFunct.write_to_txt_file(text, file_name)
    new_text = FilesFunct.read_text_from_file(file_name)
    assert new_text == text
    os.remove("test_text.txt")


def test_read_json_file_invalid_json(tmp_path):
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("not a json")
    result = FilesFunct.read_json_file(str(invalid_json_file))
    assert result is None


def test_serialization_from_files(symmetric):
    key = symmetric.generate_key(128)
    file_name = "test_key.bin"
    FilesFunct.serial_sym_key(file_name, key)
    new_key = FilesFunct.deserial_sym_key(file_name)
    assert new_key == key
    os.remove("test_key.bin")


def test_asym_public_key_serialization(asymmetric):
    public_key, private_key = asymmetric.generate_keys()
    public_key_file = "public_key.pem"
    private_key_file = "private_key.pem"
    FilesFunct.serialization_rsa_public_key(public_key, public_key_file)
    FilesFunct.serialization_rsa_private_key(private_key, private_key_file)
    new_public_key = FilesFunct.deserialization_rsa_public_key(public_key_file)
    new_private_key = FilesFunct.deserialization_rsa_private_key(
        private_key_file)
    assert new_public_key == public_key
    assert new_private_key.private_numbers() == private_key.private_numbers()
    os.remove("public_key.pem")
    os.remove("private_key.pem")
