import json
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)


def load_rsa_settings(config_path: str = "config.json") -> dict[str, Any]:
    """
    Loads RSA settings from configuration file.
    """
    try:
        with open(config_path, encoding="utf-8-sig") as f:
            config = json.load(f)
        return {
            "key_size": config.get("rsa_key_size", 2048),
            "public_exponent": config.get("rsa_public_exponent", 65537),
        }
    except Exception as e:
        print(f"Warning: Could not load RSA settings from config: {e}")
        return {"key_size": 2048, "public_exponent": 65537}


def generate_rsa_keys() -> tuple[RSAPrivateKey, RSAPublicKey]:
    """
    Generates RSA key pair using settings from config file.
    """
    settings = load_rsa_settings()

    private_key = rsa.generate_private_key(
        public_exponent=settings["public_exponent"],
        key_size=settings["key_size"],
    )
    return private_key, private_key.public_key()


def load_private_key(path: str, password: bytes | None = None) -> RSAPrivateKey:
    """
    Loads RSA private key from PEM file.
    """
    try:
        with open(path, "rb") as f:
            key = load_pem_private_key(f.read(), password=password)
            # Явная проверка типа для Pyright
            if not isinstance(key, RSAPrivateKey):
                raise ValueError("Loaded key is not an RSA private key")
            return key
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Private key file not found: {path}") from e
    except Exception as e:
        raise Exception(f"Failed to load private key: {e}") from e


def load_public_key(path: str) -> RSAPublicKey:
    """
    Loads RSA public key from PEM file.
    """
    try:
        with open(path, "rb") as f:
            key = load_pem_public_key(f.read())
            # Явная проверка типа для Pyright
            if not isinstance(key, RSAPublicKey):
                raise ValueError("Loaded key is not an RSA public key")
            return key
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Public key file not found: {path}") from e
    except Exception as e:
        raise Exception(f"Failed to load public key: {e}") from e


def encrypt(data: bytes, public_key: RSAPublicKey) -> bytes:
    """
    Encrypts data using RSA-OAEP.
    """
    return public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def decrypt(data: bytes, private_key: RSAPrivateKey) -> bytes:
    """
    Decrypts data using RSA-OAEP.
    """
    return private_key.decrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
