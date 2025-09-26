import const

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os


class CamelliaCipher:
    def __init__(self, key):
        """
        Инициализация шифра Camellia
        Параметры:
        key (bytes): ключ шифрования (16, 24 или 32 байта)
        Исключения:
        ValueError: при недопустимой длине ключа
        """
        key_length = len(key) * 8
        if key_length not in const.CAMELLIA_KEY_SIZES:
            raise ValueError(
                f"Недопустимая длина ключа Camellia ({key_length} бит). "
                f"Допустимые значения: {const.CAMELLIA_KEY_SIZES}"
            )
        self.key = key

    def encrypt(self, plaintext):
        """
        Шифрование данных
        Параметры:
        plaintext (bytes): открытый текст для шифрования
        Возвращает:
        bytes: зашифрованные данные в формате IV + ciphertext
        """
        # Генерация случайного вектора инициализации
        iv = os.urandom(const.IV_SIZE)

        # Добавление паддинга PKCS7
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        # Шифрование данных
        cipher = Cipher(
            algorithms.Camellia(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return iv + ciphertext

    def decrypt(self, ciphertext):
        """
        Дешифрование данных
        Параметры:
        ciphertext (bytes): зашифрованные данные в формате IV + ciphertext
        Возвращает:
        bytes: расшифрованный текст
        """
        # Извлечение вектора инициализации
        iv = ciphertext[:const.IV_SIZE]
        actual_ciphertext = ciphertext[const.IV_SIZE:]

        # Дешифрование данных
        cipher = Cipher(
            algorithms.Camellia(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()

        # Удаление паддинга
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

        return plaintext

    @staticmethod
    def generate_camellia_key(key_size):
        """
        Генерация случайного ключа для Camellia
        Параметры:
        key_size (int): размер ключа в битах (128, 192 или 256)
        Возвращает:
        bytes: сгенерированный ключ
        Исключения:
        ValueError: при недопустимом размере ключа
        """
        if key_size not in const.CAMELLIA_KEY_SIZES:
            raise ValueError(
                f"Недопустимый размер ключа ({key_size} бит). "
                f"Допустимые значения: {const.CAMELLIA_KEY_SIZES}"
            )
        return os.urandom(key_size // 8)

