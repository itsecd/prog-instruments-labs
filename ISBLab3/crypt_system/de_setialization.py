from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    load_pem_private_key,
)
from loguru import logger as log


def serialization_public_key(path: str, public_key: rsa.RSAPublicKey) -> None:
    """
    Сериализует публичный RSA-ключ и сохраняет его в файл в формате PEM.

    :param path: Путь к файлу
    :param public_key: Публичный RSA-ключ
    :return: None
    :raises Exception: Если произошла ошибка при записи ключа в файл (например, проблемы с доступом).
    """
    try:
        log.info(f"Starting public key serialization to: {path}")
        with open(path, "wb") as file:
            key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            file.write(key_bytes)
        log.info(f"Public key successfully serialized and saved to: {path}")
    except Exception as e:
        log.error(f"Error during public key serialization to {path}: {e}")
        raise Exception(f"An error occurred during public key serialization: {e}")


def serialization_private_key(path: str, private_key: rsa.RSAPrivateKey) -> None:
    """
    Сериализует приватный RSA-ключ и сохраняет его в файл в формате PEM без шифрования.

    :param path: Путь к файлу
    :param private_key: Приватный RSA-ключ
    :return: None
    :raises Exception: Если произошла ошибка при записи ключа в файл (например, проблемы с доступом).
    """
    try:
        log.info(f"Starting private key serialization to: {path}")
        with open(path, "wb") as file:
            key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
            file.write(key_bytes)
        log.info(f"Private key successfully serialized and saved to: {path}")
    except Exception as e:
        log.error(f"Error during private key serialization to {path}: {e}")
        raise Exception(f"An error occurred during serialization of the private key: {e}")


def serialization_symmetrical(path: str, symmetrical_key: bytes) -> None:
    """
    Сохраняет симметричный ключ в файл в виде байтов.

    :param path: Путь к файлу
    :param symmetrical_key: Симметричный ключ
    :return: None
    :raises Exception: Если произошла ошибка при записи ключа в файл
    """
    try:
        log.info(f"Starting symmetric key serialization to: {path}")
        with open(path, "wb") as file:
            file.write(symmetrical_key)
        log.info(f"Symmetric key successfully serialized and saved to: {path}")
    except Exception as e:
        log.error(f"Error during symmetric key serialization to {path}: {e}")
        raise Exception(f"An error occurred while serializing the symmetric key: {e}")


def deserialization_public_key(path: str) -> rsa.RSAPublicKey:
    """
    Десериализует публичный RSA-ключ из файла в формате PEM.

    :param path: Путь к файлу, содержащему публичный ключ в формате PEM.
    :return: Публичный RSA-ключ
    :raises Exception: Если произошла ошибка при чтении или десериализации ключа
    """
    try:
        log.info(f"Starting public key deserialization from: {path}")
        with open(path, "rb") as file:
            public_bytes = file.read()
            d_public_key = load_pem_public_key(public_bytes)
        log.info(f"Public key successfully deserialized from: {path}")
        return d_public_key
    except Exception as e:
        log.error(f"Error during public key deserialization from {path}: {e}")
        raise Exception(f"An error occurred during public key deserialization: {e}")


def deserialization_private_key(path: str) -> rsa.RSAPrivateKey:
    """
    Десериализует приватный RSA-ключ из файла в формате PEM.

    :param path: Путь к файлу, содержащему приватный ключ в формате PEM.
    :return: Приватный RSA-ключ
    :raises Exception: Если произошла ошибка при чтении или десериализации ключа
    """
    try:
        log.info(f"Starting private key deserialization from: {path}")
        with open(path, "rb") as file:
            public_bytes = file.read()
            d_public_key = load_pem_private_key(public_bytes, password=None)
        log.info(f"Private key successfully deserialized from: {path}")
        return d_public_key
    except Exception as exc:
        log.error(f"Error during private key deserialization from {path}: {exc}")
        raise Exception(f"An error occurred while deserializing the private key: {exc}")


def deserialization_symmetrical(path: str) -> bytes:
    """
    Десериализует симметричный ключ

    :param path: Путь к файлу, содержащему симметричный ключ
    :return: Симметричный ключ
    :raises Exception: Если произошла ошибка при чтении файла
    """
    try:
        log.info(f"Starting symmetric key deserialization from: {path}")
        with open(path, "rb") as file:
            key_data = file.read()
        log.info(f"Symmetric key successfully deserialized from: {path}")
        return key_data
    except Exception as e:
        log.error(f"Error during symmetric key deserialization from {path}: {e}")
        raise Exception(f"An error occurred while deserializing the symmetric key: {e}")
        
