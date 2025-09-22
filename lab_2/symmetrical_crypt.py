import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class Symmetrical:
    @staticmethod
    def symmetric_encrypt_chacha20(content, symmetric_key):
        """
        A function that encrypts data using the symmetric ChaCha20 algorithm
        :param content: Data that needs to be encrypted
        :param key: The symmetric key that we use to encrypt
        :return Encrypted data
        """
        print("Шифрование в процессе...")
        nonce = os.urandom(16)
        cipher = Cipher(algorithms.ChaCha20(symmetric_key, nonce), mode=None,
                        backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(content) + encryptor.finalize()
        print("||Шифровка завершена!||")
        return ciphertext, nonce

    @staticmethod
    def symmetric_decrypt_chacha20(encrypted_content, symmetric_key, nonce):
        """
        A function that decrypts data using the symmetric ChaCha20 algorithm
        :param encrypted_content: Encrypted data
        :param key: The symmetric key that we will use to decrypt
        :param nonce: The random number used in the ChaCha20 algorithm
        :return: Decrypted data
        """
        print("Расшифровка в процессе...")
        cipher = Cipher(algorithms.ChaCha20(symmetric_key, nonce), mode=None,
                        backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(encrypted_content) + decryptor.finalize()
        print("||Данные расшифрованы!||")
        return plaintext

