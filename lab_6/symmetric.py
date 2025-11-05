import os
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class Symmetric:
    def __init__(self):
        self._iv = os.urandom(algorithms.Camellia.block_size // 8)

    @staticmethod
    def generate_symmetric_key(key_len: int) -> bytes:

        """
        Generating symmetric key
        :param key_len: key length
        :return: key
        """

        key = os.urandom(key_len // 8)

        return key

    @staticmethod
    def iv() -> bytes:

        """
        creates iv
        :return: iv
        """

        iv = os.urandom(algorithms.Camellia.block_size // 8)

        return iv

    @staticmethod
    def padding_text(text: bytes) -> bytes:

        """
        padding text
        :param text: text
        :return: padded text
        """

        padder = padding.ANSIX923(algorithms.Camellia.block_size).padder()
        padded_text = padder.update(text) + padder.finalize()

        return padded_text

    @staticmethod
    def encrypt_text(text: bytes, key: bytes, iv: bytes) -> bytes:

        """
        encrypts text
        :param text: text to encrypt
        :param key: symmetric key
        :return: encrypted text
        """

        padded_text = Symmetric.padding_text(text)

        cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        c_text = encryptor.update(padded_text) + encryptor.finalize()

        return c_text

    @staticmethod
    def decrypt_text(c_text: bytes, key: bytes, iv: bytes) -> bytes:

        """
        decrypts text
        :param c_text: text to decrypt
        :param key: symmetric key
        :return: decrypted text
        """

        cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        dc_text = decryptor.update(c_text) + decryptor.finalize()

        return Symmetric.unpadding_text(dc_text)

    @staticmethod
    def unpadding_text(dc_text: bytes) -> bytes:

        """
        unpadding text
        :param dc_text: text
        :return: unpadded text
        """

        unpadder = padding.ANSIX923(algorithms.Camellia.block_size).unpadder()
        unpadded_dc_text = unpadder.update(dc_text) + unpadder.finalize()

        return unpadded_dc_text