import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import algorithms, Cipher, modes


class FileEncryptor:

    def __init__(self, symmetric_key):

        self.symmetric_key = symmetric_key
        self.iv = os.urandom(8)

    def encrypt(self, plaintext):

        padder = padding.PKCS7(64).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(algorithms.Blowfish(self.symmetric_key), modes.CBC(self.iv))
        encryptor = cipher.encryptor()
        return encryptor.update(padded_data) + encryptor.finalize()


class FileDecryptor:

    def __init__(self, symmetric_key):
        self.symmetric_key = symmetric_key

    def decrypt(self, ciphertext, iv):

        cipher = Cipher(algorithms.Blowfish(self.symmetric_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(64).unpadder()
        return unpadder.update(decrypted_padded) + unpadder.finalize()