import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class Symmetrical:
    """
    Класс для алгоритма шифрования CAST5
    """
    @staticmethod
    def generate_key(key_len_bits: int) -> bytes:
        """
        Генерирует ключ с заданной длиной
        :param key_len_bits: длина ключа
        :return: ключ
        """
        if not (40<=key_len_bits <= 128):
            raise ValueError("Длина ключа должна быть 40-128 бит!")
        return os.urandom(key_len_bits // 8)




    @staticmethod
    def encrypt_text(key: bytes, text: str) -> bytes:
        """
        Шифрование текстового файла симметричным алгоритмом
        :param key: симметричный ключ
        :param text: текст для шифрования в виде строки
        :return: байтовая строка
        """
        if not text:
            raise ValueError("Пустой текст для шифрования")
        if not (5<=len(key) <= 16):
            raise ValueError(f"Некорректный размер ключа: {len(key)} байт")

        bytes = text.encode('utf-8')
        padder = padding.ANSIX923(64).padder()#создаёт паддер с блоком 64 бита
        padded_text = padder.update(bytes) + padder.final()#данные + заполнение

        iv = os.urandom(8)
        #  Создание объекта шифра
        cipher = Cipher(algorithms.CAST5(key), modes.CBC(iv))
        encryptor = cipher.encryptor()#создает объект-шифратор
        return iv + encryptor.update(padded_text) + encryptor.finalize()




    @staticmethod
    def decrypt_text(key: bytes, encrypted_data: bytes) -> str:
        """
        Дешифрование файла, зашифрованного симметричным алгоритмом
        :param key: симметричный ключ
        :param encrypted_data: зашифрованные данные
        :return: строка
        """
        if len(encrypted_data) < 8:
            raise ValueError("Данные слишком короткие для расшифровки")

        iv = encrypted_data[:8]#первые 8 байт - вектор инициализации
        encrypted_text=encrypted_data[8:]#всё что после - зашифрованный текст

        if not (5 <= len(key) <= 16):
            raise ValueError(f"Некорректный размер ключа: {len(key)} байт")

        cipher = Cipher(algorithms.CAST5(key), modes.CBC(iv))#создание шифра
        decryptor = cipher.decryptor()#создание дешифратора
        decrypted_padded = decryptor.update(encrypted_text) + decryptor.final()

        unpadder = padding.ANSIX923(64).unpadder()
        # Удаление заполнения
        unpadded_text=unpadder.update(decrypted_padded) + unpadder.finalize()

        return unpadded_text.decode('UTF-8')
