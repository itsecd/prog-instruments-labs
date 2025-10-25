from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    load_pem_private_key,
)


def serialization_public_key(path: str, public_key: rsa.RSAPublicKey) -> None:
    """
    Сериализует публичный RSA-ключ и сохраняет его в файл в формате PEM.

    :param path: Путь к файлу
    :param public_key: Публичный RSA-ключ
    :return: None
    :raises Exception: Если произошла ошибка при записи ключа в файл (например, проблемы с доступом).
    """
    try:
        with open(path, "wb") as file:
            file.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
    except Exception as e:
        raise Exception(f"Возникла ошибка при сериализации public key: {e}")


def serialization_private_key(path: str, private_key: rsa.RSAPrivateKey) -> None:
    """
    Сериализует приватный RSA-ключ и сохраняет его в файл в формате PEM без шифрования.

    :param path: Путь к файлу
    :param private_key: Приватный RSA-ключ
    :return: None
    :raises Exception: Если произошла ошибка при записи ключа в файл (например, проблемы с доступом).
    """
    try:
        with open(path, "wb") as file:
            file.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
    except Exception as e:
        raise Exception(f"Возникла ошибка при сериализации private key: {e}")


def serialization_symmetrical(path: str, symmetrical_key: bytes) -> None:
    """
    Сохраняет симметричный ключ в файл в виде байтов.

    :param path: Путь к файлу
    :param symmetrical_key: Симметричный ключ
    :return: None
    :raises Exception: Если произошла ошибка при записи ключа в файл
    """
    try:
        with open(path, "wb") as file:
            file.write(symmetrical_key)
    except Exception as e:
        raise Exception(f"Возникла ошибка при сериализации symmetric key: {e}")


def deserialization_public_key(path: str) -> rsa.RSAPublicKey:
    """
    Десериализует публичный RSA-ключ из файла в формате PEM.

    :param path: Путь к файлу, содержащему публичный ключ в формате PEM.
    :return: Публичный RSA-ключ
    :raises Exception: Если произошла ошибка при чтении или десериализации ключа
    """
    try:
        with open(path, "rb") as file:
            public_bytes = file.read()
            d_public_key = load_pem_public_key(public_bytes)
            return d_public_key
    except Exception as e:
        raise Exception(f"Возникла ошибка при десериализации public key: {e}")


def deserialization_private_key(path: str) -> rsa.RSAPrivateKey:
    """
    Десериализует приватный RSA-ключ из файла в формате PEM.

    :param path: Путь к файлу, содержащему приватный ключ в формате PEM.
    :return: Приватный RSA-ключ
    :raises Exception: Если произошла ошибка при чтении или десериализации ключа
    """
    try:
        with open(path, "rb") as file:
            public_bytes = file.read()
            d_public_key = load_pem_private_key(public_bytes, password=None)
            return d_public_key
    except Exception as exc:
        raise Exception(f"Возникла ошибка при десериализации private key: {exc}")


def deserialization_symmetrical(path: str) -> bytes:
    """
    Десериализует симметричный ключ

    :param path: Путь к файлу, содержащему симметричный ключ
    :return: Симметричный ключ
    :raises Exception: Если произошла ошибка при чтении файла
    """
    try:
        with open(path, "rb") as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Возникла ошибка при десериализации symmetric key: {e}")
