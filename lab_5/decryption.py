import os
from typing import Any

from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from loguru import logger
from shared import deserialize, get_symmetric_key, serialize


@logger.catch
def remove_padding(file_data: bytes, block_size: int) -> bytes:
    """
    Функция, убирающая padding из расшифрованных данных.
    Args:
        file_data (bytes): Данные
        block_size (bytes): Размер блока блочного алгоритма
    Returns:
        Исходные данные
    """
    unpadder = padding.ANSIX923(block_size).unpadder()
    unpadded_data = unpadder.update(file_data) + unpadder.finalize()
    return unpadded_data


@logger.catch
def apply_3des_decryption(data: bytes, key: bytes, mode: Any) -> bytes:
    """
    Функция, расшифровывающая данные алгоритмом 3DES в заданном режиме.
    Args:
        data (bytes): Зашифрованные данные
        key (bytes): Ключ шифрования
        mode: Режим шифрования алгоритма
    Returns:
        Расшифрованные данные
    """
    x3des = TripleDES(key)
    cipher = Cipher(x3des, mode=mode)
    decryptor = cipher.decryptor()
    dec_data = decryptor.update(data) + decryptor.finalize()
    return dec_data


def decrypt(settings: dict[str, str | int | bytes]) -> None:
    """
    Процедура, дешифрующая зашифрованный с помощью симметричного алгоритма файл
    Args:
        settings (dict): Параметры приложения
    """
    logger.debug("Getting symmetric key")
    symmetric_key = get_symmetric_key(settings["symmetric_key"], settings["secret_key"])

    enc_file_data = deserialize(settings["encrypted_file"])
    logger.debug(f"Decrypting {os.path.abspath(settings['encrypted_file'])}")

    dec_file_data = apply_3des_decryption(enc_file_data, symmetric_key, modes.CBC(settings["3des_init_vector"]))
    unpadded_data = remove_padding(dec_file_data, TripleDES.block_size)

    serialize(settings["decrypted_file"], unpadded_data)
    logger.info(f"Decryption result saved at {os.path.abspath(settings['decrypted_file'])}")