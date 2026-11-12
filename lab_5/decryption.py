from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.primitives import padding
from colorama import Fore, Back, Style, init
from shared import serialize, deserialize, get_symmetric_key
from typing import Any
import os


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
    init(autoreset=True)
    print(Fore.GREEN + "Получаем ключ.")
    symmetric_key = get_symmetric_key(settings["symmetric_key"], settings["secret_key"])

    enc_file_data = deserialize(settings["encrypted_file"])
    print("Расшифровываем", Fore.BLUE + f"{os.path.abspath(settings['encrypted_file'])}")

    dec_file_data = apply_3des_decryption(enc_file_data, symmetric_key, modes.CBC(settings["3des_init_vector"]))
    unpadded_data = remove_padding(dec_file_data, TripleDES.block_size)

    print(Fore.GREEN + "Расшифрованный текст:")
    print(Style.DIM + unpadded_data.decode('UTF-8'))

    serialize(settings["decrypted_file"], unpadded_data)
    print("Результат сохранён в", Fore.BLUE + f"{os.path.abspath(settings['decrypted_file'])}")