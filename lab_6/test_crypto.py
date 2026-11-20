from unittest.mock import patch
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from utils import (
    load_settings,
    get_asym_padding,
    read_bytes_safe,
    write_bytes_safe,
    load_private_key,
    save_public_key,
    save_private_key,
)
from keygen import generate_keys
from encrypt import encrypt_data
from decrypt import decrypt_data


def make_settings(tmp_path):
    return {
        "public_key": str(tmp_path / "public.pem"),
        "secret_key": str(tmp_path / "private.pem"),
        "symmetric_key": str(tmp_path / "camellia_key.bin"),
        "initial_file": str(tmp_path / "text.txt"),
        "encrypted_file": str(tmp_path / "encrypted_text.bin"),
        "decrypted_file": str(tmp_path / "decrypted_text.txt"),
    }


def test_load_settings_valid(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"public_key": "public.pem"}', encoding="utf-8")
    assert load_settings(str(path)) == {"public_key": "public.pem"}


def test_load_settings_errors(tmp_path):
    assert load_settings(str(tmp_path / "missing.json")) == 0

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{broken json", encoding="utf-8")
    assert load_settings(str(bad_json)) == 0

    # Передача пути к директории вызывает ошибку открытия файла (ветка except Exception)
    assert load_settings(str(tmp_path)) == 0


def test_read_and_write_bytes_safe(tmp_path):
    path = tmp_path / "data.bin"
    assert write_bytes_safe(str(path), b"\x01\x02\x03", "Ошибка записи") is True
    assert read_bytes_safe(str(path)) == b"\x01\x02\x03"
    assert read_bytes_safe(str(tmp_path / "not_found.bin")) is None


def test_write_bytes_safe_io_error():
    with patch("builtins.open", side_effect=IOError("Disk full")):
        assert write_bytes_safe("out.bin", b"123", "Ошибка записи") is False


def test_save_and_load_rsa_keys(tmp_path):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    pub_path = str(tmp_path / "public.pem")
    priv_path = str(tmp_path / "private.pem")

    assert save_public_key(public_key, pub_path) is True
    assert save_private_key(private_key, priv_path) is True
    assert load_private_key(priv_path) is not None
    assert load_private_key(str(tmp_path / "missing.pem")) is None


def test_load_private_key_invalid_format(tmp_path):
    bad_key = tmp_path / "bad.pem"
    bad_key.write_bytes(b"not a pem key")
    assert load_private_key(str(bad_key)) is None


def test_save_keys_serialization_error():
    assert save_public_key(None, "public.pem") is False
    assert save_private_key(None, "private.pem") is False


@pytest.mark.parametrize("key_size", [128, 192, 256])
def test_generate_keys_valid_sizes(tmp_path, key_size):
    settings = make_settings(tmp_path)
    generate_keys(settings, key_size_bits=key_size)

    private_key = load_private_key(settings["secret_key"])
    enc_sym_key = read_bytes_safe(settings["symmetric_key"])
    sym_key = private_key.decrypt(enc_sym_key, get_asym_padding())

    assert len(sym_key) == key_size // 8


def test_generate_keys_invalid_size(tmp_path):
    settings = make_settings(tmp_path)
    generate_keys(settings, key_size_bits=512)
    assert read_bytes_safe(settings["public_key"]) is None


def test_generate_keys_save_failure(tmp_path):
    settings = make_settings(tmp_path)
    with patch("keygen.utils.save_public_key", return_value=False):
        generate_keys(settings)

    assert not (tmp_path / "camellia_key.bin").exists()


@pytest.mark.parametrize(
    "text",
    [
        b"Hello, world!",
        "Привет, мир! Проверка шифрования.".encode("utf-8"),
        b"16_bytes_exact!!",
    ],
)
def test_encrypt_decrypt_roundtrip(tmp_path, text):
    settings = make_settings(tmp_path)
    generate_keys(settings)
    (tmp_path / "text.txt").write_bytes(text)

    encrypt_data(settings)
    decrypt_data(settings)

    assert (tmp_path / "decrypted_text.txt").read_bytes() == text


def test_encrypt_writes_iv_in_first_16_bytes(tmp_path):
    settings = make_settings(tmp_path)
    generate_keys(settings)
    (tmp_path / "text.txt").write_bytes(b"secret message")

    fixed_iv = b"0123456789ABCDEF"
    with patch("encrypt.os.urandom", return_value=fixed_iv):
        encrypt_data(settings)

    encrypted = (tmp_path / "encrypted_text.bin").read_bytes()
    assert encrypted[:16] == fixed_iv


def test_decrypt_too_short_file(tmp_path):
    settings = make_settings(tmp_path)
    generate_keys(settings)
    (tmp_path / "encrypted_text.bin").write_bytes(b"short")

    decrypt_data(settings)
    assert not (tmp_path / "decrypted_text.txt").exists()


def test_decrypt_corrupted_ciphertext_padding_error(tmp_path):
    settings = make_settings(tmp_path)
    generate_keys(settings)
    (tmp_path / "text.txt").write_bytes(b"secret data")
    encrypt_data(settings)

    corrupted = bytearray((tmp_path / "encrypted_text.bin").read_bytes())
    corrupted[-1] ^= 0xFF
    (tmp_path / "encrypted_text.bin").write_bytes(bytes(corrupted))

    decrypt_data(settings)
    assert not (tmp_path / "decrypted_text.txt").exists()


def test_encrypt_and_decrypt_missing_files(tmp_path):
    settings = make_settings(tmp_path)
    encrypt_data(settings)
    decrypt_data(settings)
    assert not (tmp_path / "encrypted_text.bin").exists()
    assert not (tmp_path / "decrypted_text.txt").exists()


def test_encrypt_and_decrypt_wrong_private_key(tmp_path):
    settings = make_settings(tmp_path)
    generate_keys(settings)
    (tmp_path / "text.txt").write_bytes(b"secret data")
    encrypt_data(settings)

    # Подменяем приватный ключ другим ключом RSA
    other_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    save_private_key(other_private, settings["secret_key"])

    decrypt_data(settings)
    assert not (tmp_path / "decrypted_text.txt").exists()

    (tmp_path / "encrypted_text.bin").unlink()
    encrypt_data(settings)
    assert not (tmp_path / "encrypted_text.bin").exists()