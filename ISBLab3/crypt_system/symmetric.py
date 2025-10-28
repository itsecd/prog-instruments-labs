from cryptography.hazmat.primitives import padding as symmetric_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
from loguru import logger as log

class Symmetric:
    @staticmethod
    def symmetrical_keygen() -> bytes:
        """
        Генерирует случайный ключ для шифрования SM4
        :return: ключ
        """
        log.info("symmetrical key generated")
        return os.urandom(16)

    @staticmethod
    def padding(text: bytes) -> bytes:
        """
        Применяет PKCS7-отступы к тексту
        :param text: текст без отступов
        :return: Текст с отступами
        """
        padding = symmetric_padding.PKCS7(128).padder()
        padded_text = padding.update(text) + padding.finalize()
        log.info("Add padding to text")
        return padded_text

    @staticmethod
    def delete_padding(text: bytes) -> bytes:
        """
        Удаляет PKCS7-отступы из текста
        :param text: Текст с отступами
        :return: Текст без отступов
        """
        unpadder = symmetric_padding.PKCS7(128).unpadder()
        unpadded_text = unpadder.update(text) + unpadder.finalize()
        log.info("Delete padding from text")
        return unpadded_text

    @staticmethod
    def text_encrypter(text: bytes, key: bytes) -> bytes:
        """
        Шифрует текст с помощью алгоритма SM4 в режиме CBC
        :param text: Текст для шифрования
        :param key: ключ шифрования
        :return: init_vec + зашифрованный текст
        """
        if len(key) != 16:
            log.error("Invalid len of symmetrical key")
            raise ValueError("Invalid len of key. Key shoud be 16 byte (128 bit)")
        init_vec = os.urandom(16)
        padded_text = Symmetric.padding(text)
        cipher = Cipher(algorithms.SM4(key), modes.CBC(init_vec))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_text) + encryptor.finalize()
        log.info("text encrypted by symmetrical algorithm")
        return init_vec + ciphertext

    @staticmethod
    def text_decrypter(ciphertext: bytes, key: bytes) -> bytes:
        """
        Расшифровывает текст с помощью алгоритма SM4 в режиме CBC
        :param ciphertext: init_vec + зашифрованный текст
        :param key: ключ расшифровки
        :return: Расшифрованный текст
        """
        if len(key) != 16:
            log.error("Invalid len of key")
            raise ValueError("Invalid len of key. Key shoud be 16 byte (128 bit)")
        if len(ciphertext) < 16:
            log.error("ciphertext is too short")
            raise ValueError("The ciphertext is too short")
        iv = ciphertext[:16]
        actual_ciphertext = ciphertext[16:]
        cipher = Cipher(algorithms.SM4(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_text = decryptor.update(actual_ciphertext) + decryptor.finalize()
        log.info("text decrypted by symmetrical algorithm")
        return Symmetric.delete_padding(padded_text)

