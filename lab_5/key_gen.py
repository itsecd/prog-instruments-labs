import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.padding import MGF1, OAEP
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from loguru import logger
from shared import serialize


@logger.catch
def generate_symmetric_key(size: int) -> bytes:
    """
    Функция, генерирующая ключ для симметричного алгоритма шифрования.
    Args:
        size (int): Размер ключа в битах
    Returns:
        Ключ в виде набора байт
    """
    return os.urandom(size // 8)


def generate_keys(settings: dict[str, str | int | bytes]) -> None:
    """
    Процедура, создающая ключи и сериализующая их
    Args:
        settings (dict): Параметры приложения
    """
    logger.debug(f"Generating {settings['3des_key_size']} bytes symmetric key")
    symmetric_key = generate_symmetric_key(settings["3des_key_size"])

    logger.debug("Generating asymmetric keys")
    keys = rsa.generate_private_key(public_exponent=settings["rsa_public_exponent"], key_size=settings["rsa_key_size"])
    private_key = keys
    public_key = keys.public_key()

    serialize(settings["secret_key"],
              private_key.private_bytes(
                  encoding=Encoding.PEM,
                  format=PrivateFormat.TraditionalOpenSSL,
                  encryption_algorithm=NoEncryption())
              )
    logger.info(f"Private key saved at {os.path.abspath(settings['secret_key'])}")

    serialize(settings["public_key"],
              public_key.public_bytes(encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo)
              )
    logger.info(f"Public key saved at {os.path.abspath(settings['public_key'])}")
    logger.debug("Encrypting symmetric key")
    encrypted_key = public_key.encrypt(symmetric_key,
        padding=OAEP(mgf=MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )

    serialize(settings["symmetric_key"], encrypted_key)
    logger.info(f"Encrypted symmetric key saved at {os.path.abspath(settings['symmetric_key'])}")
