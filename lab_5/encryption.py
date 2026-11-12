from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.primitives import padding
from colorama import Fore, Back, Style, init
from shared import serialize, deserialize, get_symmetric_key
from typing import Any
import os


def apply_padding(file_data: bytes, block_size: int) -> bytes:
    """
    Функция, применяющая padding к шифруемым данным.
    Args:
        file_data (bytes): Данные, которые нужно подогнать под размер блока
        block_size (bytes): Размер блока блочного алгоритма
    Returns:
        Данные, подходящие под шифрование блочным алгоритмом
    """
    padder = padding.ANSIX923(block_size).padder()
    padded_data = padder.update(file_data) + padder.finalize()
    return padded_data


def apply_3des_encryption(data: bytes, key: bytes, mode: Any) -> bytes:
    """
    Функция, шифрующая данные алгоритмом 3DES в заданном режиме.
    Args:
        data (bytes): Данные для шифрования
        key (bytes): Ключ шифрования
        mode: Режим шифрования алгоритма
    Returns:
        Результат шифрования алгоритма
    """
    x3des = TripleDES(key)
    cipher = Cipher(x3des, mode=mode)
    encryptor = cipher.encryptor()
    enc_data = encryptor.update(data) + encryptor.finalize()
    return enc_data


def encrypt(settings: dict[str, str | int | bytes]) -> None:
    """
    Процедура, шифрующая содержимое исходного файла симметричным алгоритмом (3DES)
    Args:
        settings (dict): Параметры приложения
    """
    init(autoreset=True)
    print(Fore.GREEN + "Получаем ключ.")
    symmetric_key = get_symmetric_key(settings["symmetric_key"], settings["secret_key"])

    print("Шифруемый файл:", Fore.BLUE + f"{os.path.abspath(settings['initial_file'])}")
    file_data = deserialize(settings["initial_file"])

    print(Fore.GREEN + "Шифруем...")
    padded_data = apply_padding(file_data, TripleDES.block_size)
    encrypted_data = apply_3des_encryption(padded_data, symmetric_key, modes.CBC(settings["3des_init_vector"]))

    serialize(settings["encrypted_file"], encrypted_data)
    print("Результат сохранён в", Fore.BLUE + f"{os.path.abspath(settings['encrypted_file'])}")
