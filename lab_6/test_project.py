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


# 7. Тест записи и чтения JSON-файла
def test_filehandler_json_read_write(tmp_path):
    json_path = tmp_path / "settings.json"
    data = {"a": "b"}
    FileHandler.write_json(json_path, data)
    loaded = FileHandler.get_json(json_path)
    assert loaded == data


# 8. Тест гибридной генерации ключей
def test_hybrid_generate_keys():
    private_key, public_key, symmetric_key = Hybrid.generate_keys()
    assert private_key is not None
    assert public_key is not None
    assert isinstance(symmetric_key, bytes)


# 9. Тест CLI режима generate с моками файлов
@patch("main.FileHandler")
@patch("main.Asymmetrical")
@patch("main.Hybrid")
def test_main_generate_mode(mock_hybrid, mock_asym, mock_filehandler, tmp_path, monkeypatch):
    mock_args = MagicMock(mode="generate", key_length=128)
    monkeypatch.setattr("main.parse_arguments", lambda: mock_args)

    mock_hybrid.generate_keys.return_value = ("priv", "pub", b"sym")
    mock_asym.encrypt_by_public_key.return_value = b"encrypted"
    mock_filehandler.serialize_public_key.return_value = None
    mock_filehandler.serialize_private_key.return_value = None
    mock_filehandler.serialize_symmetric_key.return_value = None

    monkeypatch.setattr("main.SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr("main.FileHandler.write_json", lambda f, d: None)
    main.create_default_settings_if_needed()

    main.main()


# 10. Тест CLI режима encrypt с отсутствующим файлом
def test_main_encrypt_missing_files(monkeypatch):

    args = MagicMock(mode="encrypt", key_length=128)
    monkeypatch.setattr("main.parse_arguments", lambda: args)
    settings = {
        "public_key": "no/key.pem",
        "private_key": "no/key.pem",
        "symmetric_key": "no/key.pem",
        "initial_text": "no/text.txt",
        "encrypted_text": "no/encrypted.txt"
    }
    monkeypatch.setattr("main.load_settings", lambda: settings)

    with pytest.raises(SystemExit):
        main.main()