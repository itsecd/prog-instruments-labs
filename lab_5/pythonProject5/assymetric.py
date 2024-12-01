from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes


class RSA:
    def __init__(self):
        """
           Класс, представляющий объект RSA для работы с открытым и закрытым ключами.
           public_key: Открытый ключ RSA.
           private_key: Закрытый ключ RSA.
        """
        self.public_key = None
        self.private_key = None

    def generate_key(self, key_size=2048) -> None:
        """
           Генерирует открытый и закрытый ключи RSA заданного размера.
           key_size (int) - Размер ключа RSA.
        """
        keys = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        self.public_key = keys.public_key()
        self.private_key = keys

    def encrypt_bytes(self, bytes_: bytes) -> bytes:
        """
            Шифрует байтовые данные с использованием открытого ключа RSA.
            bytes_(bytes) -  Байтовые данные для шифрования.
            bytes - Зашифрованные байтовые данные.
         """
        enc_bytes = self.public_key.encrypt(bytes_, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                 algorithm=hashes.SHA256(), label=None))
        return enc_bytes

    def decrypt_bytes(self, bytes_: bytes) -> bytes:
        """
            Расшифровывает байтовые данные с использованием закрытого ключа RSA.
            bytes_ (bytes) - Зашифрованные байтовые данные для расшифровки.
            bytes - Расшифрованные байтовые данные.
        """
        dec_bytes = self.private_key.decrypt(bytes_, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                  algorithm=hashes.SHA256(), label=None))
        return dec_bytes