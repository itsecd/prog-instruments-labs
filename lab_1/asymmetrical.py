from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from typing import Tuple

class Asymmetrical:
    """
    Класс, который работает с RSA алгоритмом
    """
    @staticmethod
    def generate_asymmetrical_keys() -> Tuple[RSAPrivateKey, RSAPublicKey]:
        """
        Генерация открытого и закрытого ключа для RSA алгоритма
        :return: пара сгенерированных ключей
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def encrypt_by_public_key(public_key: RSAPublicKey, data: bytes) -> bytes:
        """
         Шифрование симметричного ключа открытым ключом
        :param public_key: открытый ключ
        :param data: симметричный ключ в виде байтовой строки
        :return: байтовая строка
        """
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )





    @staticmethod
    def decrypt_by_private_key(
            private_key: RSAPrivateKey,
            encrypted_data: bytes
    ) -> bytes:
        """
        Дешифрование симметричного ключа с использованием закрытого ключа
        :param private_key: закрытый ключ
        :param encrypted_data: зашифрованные данные
        :return: байтовая строка
        """
        return private_key.decrypt(
         encrypted_data,
         padding.OAEP(
             mgf=padding.MGF1(algorithm=hashes.SHA256()),#плохой комментарий
             algorithm=hashes.SHA256(),
             label=None
         )
     )