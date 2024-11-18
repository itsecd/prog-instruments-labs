import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding


class TripleDES:
    """
    A class providing methods for Triple DES generation key, encryption and decryption.
    """
    def __init__(self, symmetric_key_path: str) -> None:
        """
        Initializes a new object of the class.
        Parameters:
        - symmetric_key_path (str): Path to symmetric key.
        Returns:
        None
        """
        self.symmetric_key = symmetric_key_path

    def encrypt_3des(self, key: bytes, plaintext: bytes) -> bytes:
        """
         Encrypts plaintext using Triple DES algorithm.
         Parameters:
             key (bytes): The key to use for encryption.
             plaintext (bytes): The plaintext to encrypt.
         Returns:
             bytes: The ciphertext produced by encrypting the plaintext.
         """
        padder = padding.PKCS7(algorithms.TripleDES.block_size).padder()
        padded_plaintext = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(
            algorithms.TripleDES(key),
            modes.ECB(),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        return ciphertext

    def decrypt_3des(self, key: bytes, ciphertext: bytes) -> bytes:
        """
        Decrypts ciphertext using Triple DES algorithm.
        Parameters:
            key (bytes): The key to use for decryption.
            ciphertext (bytes): The ciphertext to decrypt.
        Returns:
            bytes: The plaintext produced by decrypting the ciphertext.
        """
        cipher = Cipher(
            algorithms.TripleDES(key),
            modes.ECB(),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext)+ decryptor.finalize()
        unpadder = padding.PKCS7(algorithms.TripleDES.block_size).unpadder()
        plaintext = unpadder.update(padded_plaintext)+ unpadder.finalize()
        return plaintext

    def ask_user_length_key(self) -> int:
        """
        Ask user length key triple DES
        Returns:
            int: length key triple DES
        Raises:
            ValueError: If the user enters a key length that is not 64, 128, or 192.
        """
        while True:
            try:
                length = int(input("Enter key length (64, 128, or 192): "))
                if length in [64, 128, 192]:
                    return length
                else:
                    raise ValueError("Please enter 64, 128, or 192.")
            except ValueError as e:
                print("Invalid input.")

    def generate_3des_key(self, length: int) -> bytes:
        """
        Generates a Triple DES key of specified length.
        Parameters:
            length (int): The length of the key in bits (64, 128, or 192).
        Returns:
            bytes: The generated Triple DES key.
        """
        return os.urandom(length//8)