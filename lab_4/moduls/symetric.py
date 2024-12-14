import os
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from moduls.reader_writer import Texting


class Symetric:
    def create_sym_key() -> bytes:
        """Создаёт симметричный ключ,ничего не принимает, возвращает ключ в байтах"""
        return os.urandom(16)

    def serialize_sym(key: bytes, path: str) -> None:
        """Сериализует симметричный ключ, принимает путь и ключ в байтах, ничего не возварщает"""
        Texting.write_bytes(path, key)

    def deserialize_sym(path: str) -> bytes:
        """Выполняет десериализацию сииметричного ключа, принмиает путь, возвращает байты"""
        key = Texting.read_bytes(path)
        return key

    def encrypt_text(text: str, key: bytes) -> bytes:
        """Выполняетя шифрования текста, получает текст строкой и ключ, возвращает зашифрованный текст в байтах"""
        padder = padding.PKCS7(64).padder()
        bi_text = bytes(text, "UTF-8")
        iv = os.urandom(8)
        cipher = Cipher(algorithms.IDEA(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        padded_text = padder.update(bi_text) + padder.finalize()
        c_text = iv + encryptor.update(padded_text) + encryptor.finalize()
        return c_text

    def decode_text(c_text: bytes, key: bytes) -> str:
        """Выолняется дешифрование текста, принимается зашифрованный текст в байтах и ключ, возвращает расшифрованный текст строкой"""
        iv = c_text[:8]
        cipher_text = c_text[8:]
        cipher = Cipher(algorithms.IDEA(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        dc_text = decryptor.update(cipher_text) + decryptor.finalize()
        unpadder = padding.PKCS7(64).unpadder()
        unpadded_dc_text = unpadder.update(dc_text) + unpadder.finalize()
        return unpadded_dc_text.decode("UTF-8")

