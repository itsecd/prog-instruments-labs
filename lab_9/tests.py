import pytest

from fileworks import (
    read_json, read_str, read_bytes, str_writer, encrypted_ChaCha20_writer, load_config,
)
from symmetric import (
    symmetric_key_generation, symmetric_key_writter, text_encryption, text_decryption,
)
from assymetric import (
    asymmetric_keys_generation, asymmetric_keys_writter, get_secret_key,
    key_encription, decrypt_key,
)


def test_load_config_dev():
    cfg = load_config("lab_9/configs/config.yaml", "lab_9/configs/dev.yaml")
    assert cfg.crypto.symmetric.key_size == 32
    assert cfg.crypto.asymmetric.key_size == 1024


def test_load_config_prod():
    cfg = load_config("lab_9/configs/config.yaml", "lab_9/configs/prod.yaml")
    assert cfg.crypto.asymmetric.key_size == 4096


def test_load_config_base_merged():
    cfg = load_config("lab_9/configs/config.yaml", "lab_9/configs/dev.yaml")
    assert cfg.crypto.symmetric.nonce_size == 8


def test_read_json(tmp_path):
    path = tmp_path / "test.json"
    path.write_text('{"key": "value"}', encoding="utf-8")
    assert read_json(str(path)) == {"key": "value"}


def test_read_json_not_found():
    with pytest.raises(FileNotFoundError):
        read_json("nonexistent.json")


def test_read_str(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("hello", encoding="utf-8")
    assert read_str(str(path)) == "hello"


def test_read_str_not_found():
    with pytest.raises(FileNotFoundError):
        read_str("nonexistent.txt")


def test_read_bytes(tmp_path):
    path = tmp_path / "test.bin"
    path.write_bytes(b"\x01\x02\x03")
    assert read_bytes(str(path)) == b"\x01\x02\x03"


def test_read_bytes_not_found():
    with pytest.raises(FileNotFoundError):
        read_bytes("nonexistent.bin")


def test_str_writer(tmp_path):
    path = tmp_path / "out.txt"
    str_writer("hello", str(path))
    assert path.read_text(encoding="utf-8") == "hello"


def test_encrypted_ChaCha20_writer(tmp_path):
    path = tmp_path / "out.bin"
    nonce = b"12345678"
    text = b"secret"
    encrypted_ChaCha20_writer(text, nonce, str(path))
    data = path.read_bytes()
    assert data[:8] == nonce
    assert data[8:] == text


def test_symmetric_key_generation():
    key = symmetric_key_generation()
    assert isinstance(key, bytes)
    assert len(key) == 32


def test_symmetric_key_writter(tmp_path):
    path = tmp_path / "key.bin"
    key = b"a" * 32
    symmetric_key_writter(str(path), key)
    assert path.read_bytes() == key


def test_text_encryption_decryption_roundtrip():
    key = symmetric_key_generation()
    text = "Hello, world! Это тест."
    encrypted, nonce = text_encryption(text, key)
    assert len(nonce) == 8
    decrypted = text_decryption(nonce + encrypted, key)
    assert decrypted == text


def test_text_encryption_invalid_key_length():
    with pytest.raises(ValueError, match="Key length must be 32"):
        text_encryption("hello", b"short")


def test_text_decryption_invalid_key_length():
    with pytest.raises(ValueError, match="Key length must be 32"):
        text_decryption(b"\x00" * 10, b"short")


def test_text_decryption_too_short():
    key = symmetric_key_generation()
    with pytest.raises(ValueError, match="Encrypted text length must higher than 8"):
        text_decryption(b"\x00" * 5, key)


def test_asymmetric_keys_generation():
    private, public = asymmetric_keys_generation()
    assert private is not None
    assert public is not None
    assert private.public_key() == public


def test_asymmetric_keys_writter(tmp_path):
    secret_path = tmp_path / "secret.pem"
    public_path = tmp_path / "public.pem"
    private, public = asymmetric_keys_generation()
    asymmetric_keys_writter(str(secret_path), str(public_path), (private, public))
    assert secret_path.exists()
    assert public_path.exists()
    loaded = get_secret_key(str(secret_path))
    assert loaded is not None


def test_key_encription_fallback_on_too_long_key():
    _, public_key = asymmetric_keys_generation()
    too_long_key = b"x" * 300 
    result = key_encription(too_long_key, public_key)
    assert isinstance(result, bytes)
    assert len(result) == 256


def test_key_encription_decrypt_roundtrip():
    private, public = asymmetric_keys_generation()
    sym_key = symmetric_key_generation()
    encrypted = key_encription(sym_key, public)
    decrypted = decrypt_key(encrypted, private)
    assert decrypted == sym_key


def test_decrypt_key_invalid_data():
    private, _ = asymmetric_keys_generation()
    bad_data = b"\x00" * 256
    result = decrypt_key(bad_data, private)
    assert result == bad_data