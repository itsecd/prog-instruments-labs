import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class SymmetricCipher:
    def __init__(self, key):
        """
        Initializes the symmetric cipher with provided key

        :param key: Symmetric key for encryption/decryption
        """
        self.key = key
        self.backend = default_backend()

    def encrypt(self, data):
        """
        Encrypts data using Camellia algorithm in CBC mode

        :param data: Plaintext data to encrypt
        :return: Encrypted data
        """
        try:
            iv = os.urandom(16)
            cipher = Cipher(algorithms.Camellia(self.key), modes.CBC(iv), backend=self.backend)
            encryptor = cipher.encryptor()
            if len(data) % 16 != 0:
                data += b' ' * (16 - len(data) % 16)
            return iv + encryptor.update(data) + encryptor.finalize()
        except Exception as e:
            raise Exception(f"Error at encrypting symmetric key: {str(e)}")

    def decrypt(self, data):
        """
        Decrypts data using Camellia algorithm in CBC mode

        :param data: Encrypted data
        :return: Decrypted plaintext data
        """
        try:
            iv = data[:16]
            cipher = Cipher(algorithms.Camellia(self.key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()
            return decryptor.update(data[16:]) + decryptor.finalize()
        except Exception as e:
            raise Exception(f"Error at decrypting symmetric key: {str(e)}")
