import os

from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def generate_symmetric_key(key_size_bytes: int) -> bytes:
    """
    Генерирует случайный симметричный ключ AES заданного размера.

    Parameters
    ----------
    key_size_bytes : int
        Длина ключа в байтах (16, 24 или 32 для AES).

    Returns
    -------
    bytes
        Сгенерированный криптографически стойкий ключ.
    """
    if key_size_bytes not in (16, 24, 32):
        raise ValueError("AES key must be 16, 24 or 32 bytes")

    return os.urandom(key_size_bytes)


def encrypt(key: bytes, data: bytes) -> bytes:
    """
    Шифрует данные алгоритмом AES в режиме CBC с PKCS7 padding.

    Parameters
    ----------
    key : bytes
        Симметричный ключ AES.
    data : bytes
        Исходные данные для шифрования.

    Returns
    -------
    bytes
        IV (16 байт) + зашифрованный текст.
    """
    iv = os.urandom(16)

    padder = sym_padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv + ciphertext


def decrypt(key: bytes, data: bytes) -> bytes:
    """
    Расшифровывает данные AES-CBC и удаляет PKCS7 padding.

    Parameters
    ----------
    key : bytes
        Симметричный ключ AES.
    data : bytes
        Данные формата IV + ciphertext.

    Returns
    -------
    bytes
        Исходные расшифрованные данные.
    """
    iv = data[:16]
    ciphertext = data[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()

    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = sym_padding.PKCS7(128).unpadder()
    return unpadder.update(padded_data) + unpadder.finalize()
