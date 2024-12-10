import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class Symmetric:
    """
    Class for implementing symmetric algorithms
    """
    def __init__(self) -> None:
        self.key = None

    def generate_key(self) -> bytes:
        """
        Generate a random key for symmetric encryption.
        """
        self.key = os.urandom(16)
        return self.key
    
    def serialize_key(self, key_path: str) -> None:
        """
        Serialize the symmetric key and save it to a file.
        """
        with open(key_path, "wb") as key_file:
            key_file.write(self.key)

    def deserialize_key(self, key_path: str) -> None:
        """
        Deserialize the symmetric key from a file.
        """
        with open(key_path, mode='rb') as key_file: 
            self.key = key_file.read()

    def encrypt_text(self, text: bytes) -> bytes:
        """
        Encrypt text using AES symmetric encryption in CBC mode.
        """
        iv = os.urandom(8)
        cipher = Cipher(algorithms.IDEA(self.key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(64).padder()
        padded_text = padder.update(text) + padder.finalize()
        return iv + encryptor.update(padded_text) + encryptor.finalize()
    
    def decrypt_text(self, text: bytes) -> str:
        """
        Decrypt text encrypted using AES symmetric encryption in CBC mode.
        """
        iv = text[:8]
        cipher_text = text[8:]
        cipher = Cipher(algorithms.IDEA(self.key), modes.CBC(iv))
        decrypt = cipher.decryptor()
        unpacker_text = decrypt.update(cipher_text) + decrypt.finalize()
        decrypt_text = unpacker_text.decode('UTF-8')
        return decrypt_text