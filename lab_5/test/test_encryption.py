import pytest
import os

from cryptography.hazmat.primitives.asymmetric import rsa

from files import FilesHelper
from symmetric import Symmetric
from asymmetric import Asymmetric


def create_file(data: bytes, extention: str=".txt") -> str:
    file = f"temp_file_{extention}"
    with open (file,  "wb") as f:
        f.write(data)
    return file


@pytest.fixture
def symmetric_key():
    key = os.urandom(16)
    return key


@pytest.fixture
def asymmetric_keys():
    private_key = rsa.generate_private_key(
        public_exponent = 65537,
        key_size = 2048
    )
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def symmetric_cipher():
    return Symmetric()


@pytest.mark.parametrize("path", ["setings.json", "some.json"])
def test_get_json(path):
    with pytest.raises(Exception):
        FilesHelper.get_json(path)


def test_serialize_sym_key(symmetric_key):
    FilesHelper.write_bytes("symmetric_key.txt", symmetric_key)
    data = FilesHelper.get_bytes("symmetric_key.txt")
    assert data == symmetric_key
    os.remove("symmetric_key.txt")


def test_generation_sym_key_valid(symmetric_cipher):
    key = symmetric_cipher.generation_symmetric_key(16)
    assert len(key) == 16

 
@pytest.mark.parametrize("invalid_key_size", [8, 0, 24, 32])
def test_generation_sym_key_invalid(symmetric_cipher, invalid_key_size):
    with pytest.raises(ValueError) as e:
        symmetric_cipher.generation_symmetric_key(invalid_key_size)
    assert "SEED algorithm requires a 128-bit (16-bytes) key" in str(e.value)


def test_key_serialization(symmetric_cipher):
    key_size = 16
    key = symmetric_cipher.generation_symmetric_key(key_size)
    key_file = "key.bin"
    symmetric_cipher.serialization_symmetric_key(key_file)
    deserialized_key = symmetric_cipher.deserialization_symmetric_key(key_file)
    assert key == deserialized_key
    os.remove("key.bin")


@pytest.mark.parametrize("text", [
    b"Some super-secret message",
    b"",
    b"1005036312"*10,
    b"Some importnt message with different symbols `~!@#$%^&*()_-=+[]{}\|;:'\",./<>?"
    ])
def test_encrypted_text(symmetric_cipher, text):
    key_size = 16
    key = symmetric_cipher.generation_symmetric_key(key_size)

    text_file = create_file(text)
    key_file = create_file(key, ".key")
    encrypted_file = "encrypted.txt"

    encrypted_text = symmetric_cipher.encrypted_text(text_file, encrypted_file, key_file)
    
    assert len(encrypted_text) > len(text)

    os.remove(text_file)
    os.remove(key_file)
    os.remove(encrypted_file)


@pytest.mark.parametrize("text", [
    b"Some super-secret message",
    b"",
    b"1005036312"*10,
    b"Some importnt message with different symbols `~!@#$%^&*()_-=+[]{}\|;:'\",./<>?"
    ])
def test_decrypted_text(symmetric_cipher, text):
    key_size = 16
    key = symmetric_cipher.generation_symmetric_key(key_size)

    text_file = create_file(text)
    key_file = create_file(key, ".key")
    encrypted_file = "encrypted.txt"
    decrypted_file = "decrypted.txt"

    symmetric_cipher.encrypted_text(text_file, encrypted_file, key_file)
    symmetric_cipher.decrypted_text(encrypted_file, key_file, decrypted_file)

    with open(text_file, "rb") as f:
        original_text = f.read()
    
    with open(decrypted_file, "rb") as f:
        decrypted = f.read()

    assert decrypted.decode("utf-8") == original_text.decode("utf-8")

    os.remove(text_file)
    os.remove(key_file)
    os.remove(encrypted_file)
    os.remove(decrypted_file)
