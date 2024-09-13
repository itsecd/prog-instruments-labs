import logging
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key


def load_settings(setting_file: str) -> dict:
    settings = None
    try:
        with open(setting_file) as f:
            settings = json.load(f)
        logging.info("Successfully reading the settings")
    except Exception as e:
        logging.error(f"Error reading setttings file: {e}")
    return settings


def write_symmetric_key(key: bytes, filename: str) -> None:
    try:
        with open(filename, "wb") as f:
            f.write(key)
        logging.info(f"The symmetric key is written in the file {filename}")
    except Exception as e:
        logging.error(f"Error writing symmetric key to file: {e}")


def load_symmetric_key(filename: str) -> bytes:
    try:
        with open(filename, mode="rb") as f:
            content = f.read()
        logging.info(f"The symmetric key is read from the file {filename}")
    except Exception as e:
        logging.error(f"Error reading symmetric key file: {e}")
    return content


def write_asymmetric_key(
        private_key: rsa.RSAPrivateKey,
        public_key: rsa.RSAPublicKey,
        private_pem: str,
        public_pem: str
) -> None:
    try:
        with open(public_pem, "wb") as public_out:
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            public_out.write(public_bytes)
        logging.info(f"The public key successfully {public_pem}")

        with open(private_pem, "wb") as private_out:
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )
            private_out.write(private_bytes)
        logging.info(f"The private key successfully {private_pem}")
    except Exception as e:
        logging.error(f"Error writing asymmetric keys to files: {e}")


def load_private_key(filename: str) -> bytes:
    try:
        with open(filename, mode="rb") as f:
            private_bytes = f.read()
        d_private_bytes = load_pem_private_key(private_bytes, password=None,)
        logging.info(f"The private key is read from the file {filename}")
        return d_private_bytes
    except Exception as e:
        logging.error(f"Error reading private key file: {e}")


def load_text(filename: str) -> bytes:
    try:
        with open(filename, mode="rb") as f:
            text = f.read()
        logging.info(f" File {filename} readed")
        return text
    except Exception as e:
        logging.error(f"Error reading text file {e}")


def write_file(filename: str, text: bytes) -> None:
    try:
        with open(filename, mode="wb") as f:
            f.write(text)
        logging.info(f"The text is written to a file {filename}")
    except Exception as e:
        logging.error(f"Error writing text to file: {e}")