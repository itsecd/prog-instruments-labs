import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from files import FilesHelper


class Symmetric:
    """
    Class for working with a symmetric encryption algorithm
    """
    def __init__(self):
        """Constructor"""
        self.key = None

    def generation_symmetric_key(self) -> bytes:
        """
        Key generation of a symmetric encryption algorithm

        Return:
                bytes: key
        """
        self.key = os.urandom(16)
        return self.key

    def serialization_symmetric_key(self, file_name: str) -> None:
        """
        Serializing the symmetric algorithm key to a file

        Args: 
                file_name: path to the file to write
        """
        FilesHelper.write_bytes(file_name, self.key)

    @staticmethod
    def deserialization_symmetric_key(file_name: str) -> bytes:
        """
        Deserialization of the symmetric algorithm key

        Args:
                file_name: path to the file

        Return: 
                bytes: symmetric key
        """
        key = FilesHelper.get_bytes(file_name)
        return key

    def encrypted_text(self, file_name: str, encryption_path: str, key: bytes) -> bytes:
        """
        Encrypting text with a symmetric algorithm and writing to a file

        Args:
                 file_name: path to the original text file 
                 encryption_path: path to the encrypted text file
                 key: path to the symmetric key file

        Return:
                 bytes: encrypted text
        """
        padder = padding.PKCS7(128).padder()
        text = FilesHelper.get_bytes(file_name)
        padded_text = padder.update(text) + padder.finalize()

        iv = os.urandom(16)
        symmetric_key = self.deserialization_symmetric_key(key)
        cipher = Cipher(algorithms.SEED(symmetric_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        c_text = encryptor.update(padded_text) + encryptor.finalize()
        c_text = iv + c_text
        FilesHelper.write_bytes(encryption_path, c_text)

        return c_text

    def decrypted_text(self, encryption_path: str, key: bytes, decryptor_path: str) -> str:
        """
        Decryption and depadding of text using a symmetric algorithm

        Args:
                encryption_path: path to the encrypted text file
                key: path to the symmetric key file
                decryptor_path: path to the decrypted text file

        Return:
                decrypted text
        """
        en_text = FilesHelper.get_bytes(encryption_path)

        iv = en_text[:16]
        key = self.deserialization_symmetric_key(key)
        cipher = Cipher(algorithms.SEED(key), modes.CBC(iv))

        en_text = en_text[16:]

        decryptor = cipher.decryptor()
        dc_text = decryptor.update(en_text) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        unpadder_dc_text = unpadder.update(dc_text) +  unpadder.finalize()

        FilesHelper.write_txt(decryptor_path, unpadder_dc_text.decode('UTF-8'))

        return unpadder_dc_text.decode('UTF-8')
  