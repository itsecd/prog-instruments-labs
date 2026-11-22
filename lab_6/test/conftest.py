import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import ConfigLoader


@pytest.fixture
def config():
    return {
        "paths": {
            "keys_directory": "keys",
            "private_rsa_key": "private_rsa.pem",
            "public_rsa_key": "public_rsa.pem",
            "encrypted_output": "encrypted_session_key.bin",
            "input_text_file": "plaintext.txt",
            "output_encrypted_file": "ciphertext.bin",
            "key_names": {
                "private_rsa": "private_rsa_key",
                "public_rsa": "public_rsa_key",
            },
        },
        "crypto": {
            "seed": [16, 16],
            "chacha20": [32, 12],
            "rsa": [2048, 65537, ["OAEP", "SHA256", "SHA256", None]],
        },
        "validation": {"rsa": [2048, 16384, 3, 65537]},
        "ui": {
            "available_algorithms": ["SEED", "ChaCha20"],
            "window_title": "Гибридная криптосистема",
            "window_size": "600x500",
        },
    }


@pytest.fixture
def config_file(tmp_path, config):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return str(p)


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Сбрасывает Singleton, чтобы тесты не влияли друг на друга."""
    ConfigLoader._instance = None
    ConfigLoader._config = None
    yield
    ConfigLoader._instance = None
    ConfigLoader._config = None