import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


class SymmetricCrypto:
    """Класс для работы с симметричным шифрованием (AES)"""

    @staticmethod
    def generate_key(key_size: int = 256) -> bytes:
        """Генерирует симметричный ключ"""
        if key_size not in [128, 192, 256]:
            raise ValueError("Key size must be 128, 192 or 256 bits")
        return os.urandom(key_size // 8)

    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> bytes:
        """Шифрует данные с использованием AES в режиме CBC"""
        # Генерация IV
        iv = os.urandom(16)

        # Добавление padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()

        # Шифрование
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        return iv + encrypted_data

    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
        """Дешифрует данные с использованием AES в режиме CBC"""
        # Извлечение IV
        iv = encrypted_data[:16]
        actual_encrypted_data = encrypted_data[16:]

        # Дешифрование
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_data = decryptor.update(actual_encrypted_data) + decryptor.finalize()

        # Удаление padding
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

        return data