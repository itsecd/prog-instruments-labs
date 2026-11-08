import os

from cryptography.hazmat.primitives import padding as symmetric_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from file_utils import read_binary_file, write_binary_file


def generate_symmetric_key(key_size: int) -> bytes:
    """
    Генерирует ключ Camellia.

    :param key_size: длина ключа в битах, 128, 192 или 256
    :return: ключ Camellia
    """
    if key_size not in (128, 192, 256):
        raise ValueError("Длина ключа Camellia должна быть 128, 192 или 256 бит.")

    return os.urandom(key_size // 8)


def encrypt_file_camellia(input_file_path: str, output_file_path: str, key: bytes) -> None:
    """
    Шифрует файл алгоритмом Camellia-CBC.

    :param input_file_path: путь к исходному файлу
    :param output_file_path: путь для сохранения зашифрованного файла
    :param key: ключ Camellia
    :return: None
    """
    plaintext = read_binary_file(input_file_path)

    padder = symmetric_padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()

    iv = os.urandom(16)
    cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))

    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()

    write_binary_file(output_file_path, iv + ciphertext)


def decrypt_file_camellia(input_file_path: str, output_file_path: str, key: bytes) -> None:
    """
    Дешифрует файл алгоритмом Camellia-CBC.

    :param input_file_path: путь к зашифрованному файлу
    :param output_file_path: путь для сохранения расшифрованного файла
    :param key: ключ Camellia
    :return: None
    """
    encrypted_data = read_binary_file(input_file_path)

    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]

    cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = symmetric_padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

    write_binary_file(output_file_path, plaintext)
