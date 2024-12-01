import os
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.decrepit.ciphers.algorithms import CAST5


class CAST_5:
    def __init__(self, key=None):
        """
            Класс, представляющий алгоритм CAST-5 для шифрования данных.
            key - Ключ для шифрования данных.
        """
        self.key = key

    def generate_key(self, key_size=16) -> None:
        """
            Генерирует случайный ключ заданного размера и сохраняет его в атрибут key.
            key_size (int) - Размер ключа в байтах (по умолчанию 16).
        """
        self.key = os.urandom(key_size)

    def encrypt_bytes(self, bytes_: bytes) -> bytes:
        padder = padding.ANSIX923(128).padder()
        padded_bytes = padder.update(bytes_) + padder.finalize()

        iv = os.urandom(8)
        cipher = Cipher(CAST5(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        c_bytes = encryptor.update(padded_bytes) + encryptor.finalize()

        return iv + c_bytes

    def decrypt_bytes(self, bytes_: bytes) -> bytes:
        iv = bytes_[:8]
        ciphertext = bytes_[8:]

        cipher = Cipher(CAST5(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        res = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = padding.ANSIX923(128).unpadder()
        unpadded_bytes = unpadder.update(res) + unpadder.finalize()
        return unpadded_bytes
