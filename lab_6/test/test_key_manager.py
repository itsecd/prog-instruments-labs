import os
import pytest
from crypto.asymmetric import AsymmetricCrypto
from utils.file_utils import FileManager
from utils.key_manager import KeyManager


def test_ensure_creates_and_idempotent(config, tmp_path):
    """Проверяет создание RSA-ключей и то, что повторный вызов не перезаписывает их."""
    config["paths"]["keys_directory"] = str(tmp_path / "keys")
    fm = FileManager(config)
    km = KeyManager(
        config,
        fm,
        fm.get_path("private_rsa_key"),
        fm.get_path("public_rsa_key"),
    )

    assert not os.path.exists(km._private_path)
    assert not os.path.exists(km._public_path)

    km.ensure_rsa_keys_exist()
    assert os.path.exists(km._private_path)
    assert os.path.exists(km._public_path)

    with open(km._private_path, "rb") as f:
        first = f.read()
    km.ensure_rsa_keys_exist()
    with open(km._private_path, "rb") as f:
        second = f.read()
    assert first == second

    public = km.load_public_key()
    private = km.load_private_key()
    asym = AsymmetricCrypto(config)
    enc = asym.encrypt_with_public_key(b"data", public)
    assert asym.decrypt_with_private_key(enc, private) == b"data"