import json
import pytest
import logging
import os

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from moduls.reader_writer import Texting
from moduls.asymetric import Asymetric
from moduls.symetric import Symetric


def test_read_json_file():
    test_data = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f)
    result = Texting.read_json_file("test.json")
    assert result == test_data
    os.remove("test.json")

def test_write_bytes():
    path = "test_file.bin"
    key = b"the_best_string_for_testing"
    Texting.write_bytes(path, key)
    result = Texting.read_bytes(path)
    assert result == key
    os.remove(path)

def test_write_file():
    path = "test_file.txt"
    text = "Лучшая строчка для тестов."
    Texting.write_file(path, text)
    result = Texting.read_file(path)
    assert result == text
    os.remove(path)

def test_create_sym_key_type():
    key = Symetric.create_sym_key()
    assert isinstance(key, bytes) and len(key) == 16

def test_create_asym_key_type():
    keys = Asymetric.create_asym_key()
    assert isinstance(keys["private"], rsa.RSAPrivateKey) and keys["private"].key_size == 1024
    assert isinstance(keys["public"], rsa.RSAPublicKey)

def test_encrypt_decrypt_sym_key():
    asym_key = Asymetric.create_asym_key()
    sym_key = Symetric.create_sym_key()

    encrypted_key = Asymetric.encrypt_sym_key(asym_key, sym_key)
    decrypted_key = Asymetric.decrypt_sym_key(asym_key, encrypted_key)

    assert sym_key == decrypted_key

def test_encrypt_decrypt_text():
    key = Symetric.create_sym_key()
    text = "Супер секретное сообщение"
    encrypted_text = Symetric.encrypt_text(text, key)
    decrypted_text = Symetric.decode_text(encrypted_text, key)
    assert text == decrypted_text

def test_de_serialize_private_public_key(): 

    """По скольку до и после сериализации ключи могут отличаться
    пришлось делать такую сложную проверку на их совпадение"""

    asym_key = Asymetric.create_asym_key()
    private_path = "test_private_file.pem"
    public_path = "test_public_file.pem"
    Texting.serialize_private(asym_key, private_path)
    Texting.serialize_public(asym_key, public_path)
    restored_keys = Asymetric.deserylie_asym(public_path, private_path)

    assert restored_keys["public"].public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ) == asym_key["public"].public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    
    assert restored_keys["private"].private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ) == asym_key["private"].private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    os.remove(private_path)
    os.remove(public_path)