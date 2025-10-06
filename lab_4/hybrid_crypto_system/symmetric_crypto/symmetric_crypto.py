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
        return os.urandom(key_length // BYTES)

    @staticmethod
    def get_iv() -> bytes:
        return os.urandom(BYTES)

    @staticmethod
    def padding(text: bytes) -> bytes:
        padder = padding.ANSIX923(TripleDES.block_size).padder()
        return padder.update(text) + padder.finalize()

    @staticmethod
    def unpadding(text: bytes) -> bytes:
        unpadder = padding.ANSIX923(TripleDES.block_size).unpadder()
        return unpadder.update(text) + unpadder.finalize()

    @staticmethod
    def split_from_iv(text: bytes) -> tuple:
        iv = text[-BYTES:]
        split_text = text[:-BYTES]
        return iv, split_text

    @staticmethod
    def encrypt_data(plain_text, private_bytes, encrypted_symmetric_key):
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
    def decrypt_data(encrypted_data, private_bytes, encrypted_symmetric_key):
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
