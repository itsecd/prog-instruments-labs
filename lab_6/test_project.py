import os
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from asymmetrical import Asymmetrical
from symmetrical import Symmetrical
from filehandler import FileHandler
from hybrid import Hybrid
import main


# 1. Тест генерации RSA-ключей
def test_generate_asymmetrical_keys():
    private_key, public_key = Asymmetrical.generate_asymmetrical_keys()
    assert private_key is not None
    assert public_key is not None
    assert private_key.key_size == 2048


# 2. Тест генерации симметричных ключей
@pytest.mark.parametrize("bits", [40, 64, 128])
def test_generate_symmetrical_key_valid(bits):
    key = Symmetrical.generate_key(bits)
    assert isinstance(key, bytes)
    assert len(key) == bits // 8


# 3. Тест некорректной длины ключей
@pytest.mark.parametrize("bits", [0, 39, 129])
def test_generate_symmetrical_key_invalid(bits):
    with pytest.raises(ValueError):
        Symmetrical.generate_key(bits)


# 4. Тест шифрования и расшифровки симметричным алгоритмом
def test_symmetrical_encrypt_decrypt_roundtrip():
    key = Symmetrical.generate_key(128)
    text = "Hello, CAST5!"
    encrypted = Symmetrical.encrypt_text(key, text)
    decrypted = Symmetrical.decrypt_text(key, encrypted)
    assert decrypted == text


# 5. Проверка, что пустой текст вызывает ошибку
def test_encrypt_empty_text_error():
    key = Symmetrical.generate_key(128)
    with pytest.raises(ValueError):
        Symmetrical.encrypt_text(key, "")


# 6. Тест сериализации и десериализации ключей
def test_filehandler_serialize_deserialize_keys(tmp_path):
    private_key, public_key = Asymmetrical.generate_asymmetrical_keys()
    priv_path = tmp_path / "priv.pem"
    pub_path = tmp_path / "pub.pem"

    FileHandler.serialize_private_key(priv_path, private_key)
    FileHandler.serialize_public_key(pub_path, public_key)

    loaded_priv = FileHandler.deserialization_private_key(priv_path)
    loaded_pub = FileHandler.deserialization_public_key(pub_path)

    assert loaded_priv.key_size == private_key.key_size
    assert loaded_pub.public_numbers() == public_key.public_numbers()
