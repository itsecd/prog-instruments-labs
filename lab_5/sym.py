from Crypto.Cipher import CAST
from Crypto.Util.Padding import pad, unpad
import os
import logging

logger = logging.getLogger(__name__)


def create_symmetric_components(key_size: int) -> bytes:
    """
    Генерирует ключ для алгоритма CAST5 с заданным размером.
    :param key_size: Размер ключа
    :return: Ключ в бинарном формате
    """
    if not (40 <= key_size <= 128) or key_size % 8 != 0:
        error_msg = f"Некорректный размер ключа (должен быть 40-128 бит, кратно 8): {key_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    key = os.urandom(key_size // 8)
    logger.debug(f"Сгенерирован симметричный ключ размером {key_size} бит")
    return key


def encrypt_with_cast5(key: bytes, plaintext: str) -> bytes:
    """
    Шифрует текст алгоритмом CAST5.
    :param key: Ключ шифрования
    :param plaintext: Текст для шифрования
    :return: Зашифрованные данные (IV + ciphertext)
    """
    logger.debug(f"Шифрование CAST5, размер текста: {len(plaintext)} символов")
    iv = os.urandom(8)
    cipher = CAST.new(key, CAST.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext.encode('utf-8'), 8))
    result = iv + ciphertext
    logger.debug(f"Шифрование завершено, размер результата: {len(result)} байт")
    return result


def decrypt_with_cast5(key: bytes, ciphertext: bytes) -> str:
    """
    Расшифровывает данные алгоритмом CAST5.
    :param key: Ключ шифрования
    :param ciphertext: Данные для расшифровки (IV + ciphertext)
    :return: Расшифрованный текст
    """
    try:
        logger.debug(f"Дешифрование CAST5, размер данных: {len(ciphertext)} байт")
        iv = ciphertext[:8]
        actual_ciphertext = ciphertext[8:]
        cipher = CAST.new(key, CAST.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(actual_ciphertext), 8)
        result = decrypted.decode('utf-8')
        logger.debug(f"Дешифрование завершено, размер текста: {len(result)} символов")
        return result
    except Exception as e:
        error_msg = f"Ошибка дешифрования: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)