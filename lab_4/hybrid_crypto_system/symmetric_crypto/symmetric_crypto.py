from typing import Tuple
import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES

from hybrid_crypto_system.asymmetric_crypto.asymmetric_crypto import AsymmetricCrypto
from hybrid_crypto_system.de_serialization.de_serialization import DeSerialization
from hybrid_crypto_system.symmetric_crypto.constants import BYTES


class SymmetricCrypto:

    @staticmethod
    def generate_key(key_length: int) -> bytes:
        """
        Method to generate symmetric key
        :param key_length: key size
        :return: symmetric key
        """
        return os.urandom(key_length // BYTES)

    @staticmethod
    def get_iv() -> bytes:
        """
        Method to get iv
        :return: iv
        """
        return os.urandom(BYTES)

    @staticmethod
    def padding(text: bytes) -> bytes:
        """
        Method to padding data
        :param text: text to padding
        :return: padded text
        """
        padder = padding.ANSIX923(TripleDES.block_size).padder()
        return padder.update(text) + padder.finalize()

    @staticmethod
    def unpadding(text: bytes) -> bytes:
        """
        Method to unpadding data
        :param text: text to unpadding
        :return: unpadded text
        """
        unpadder = padding.ANSIX923(TripleDES.block_size).unpadder()
        return unpadder.update(text) + unpadder.finalize()

    @staticmethod
    def split_from_iv(text: bytes) -> Tuple[bytes, bytes]:
        """
        Method splits data and iv
        :param text: text to split
        :return: iv, split text
        """
        iv = text[-BYTES:]
        split_text = text[:-BYTES]
        return iv, split_text

    @staticmethod
    def encrypt_data(
            plain_text: bytes,
            private_bytes: bytes,
            encrypted_symmetric_key: bytes
    ) -> bytes:
        """
        Method to encrypt data from plain_text
        :param plain_text: text to encrypt
        :param private_bytes: private key to decrypt key
        :param encrypted_symmetric_key: key to encrypt data
        :return: encrypted data
        """
        private_key = DeSerialization.deserialization_rsa_key(private_bytes, "private")
        symmetric_key = AsymmetricCrypto.decrypt_symmetric_key(
            encrypted_symmetric_key, private_key
        )

        iv = SymmetricCrypto.get_iv()
        padded_text = SymmetricCrypto.padding(plain_text)

        cipher = Cipher(TripleDES(symmetric_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        encrypted_text = encryptor.update(padded_text) + encryptor.finalize()
        return encrypted_text + iv

    @staticmethod
    def decrypt_data(
            encrypted_data: bytes,
            private_bytes: bytes,
            encrypted_symmetric_key: bytes
    ) -> bytes:
        """
        Method to decrypt data from encrypted data
        :param encrypted_data: encrypted text
        :param private_bytes: private key as bytes
        :param encrypted_symmetric_key: symmetric key as bytes
        :return: decrypted data
        """
        private_key = DeSerialization.deserialization_rsa_key(private_bytes, "private")
        symmetric_key = AsymmetricCrypto.decrypt_symmetric_key(
            encrypted_symmetric_key, private_key
        )

        iv, text = SymmetricCrypto.split_from_iv(encrypted_data)

        cipher = Cipher(TripleDES(symmetric_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_text = decryptor.update(text) + decryptor.finalize()

        decrypted_text = SymmetricCrypto.unpadding(padded_text)
        return decrypted_text
