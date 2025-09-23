from asymmetrical import Asymmetrical
from symmetrical import Symmetrical
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

class Hybrid:
    """
    Класс совмещающий RSA и CAST5 алгоритмы
    """
    @staticmethod
    def generate_keys(key_length: int = 128) -> tuple:
        """
        Генерация и сохранение ключей для гибридной криптосистемы
        :param key_length: длина ключа
        :return: кортеж из трёх ключей
        """
        private_key, public_key = Asymmetrical.generate_asymmetrical_keys()
        symmetric_key = Symmetrical.generate_key(key_length)
        return private_key, public_key, symmetric_key

    @staticmethod
    def encrypt_data(
            private_key: RSAPrivateKey,
            encrypted_sym_key: bytes,
            plaintext: str) -> bytes:
        """
        Шифрование данных с помощью гибридного шифрования
        :param private_key: закрытый RSA ключ
        :param encrypted_sym_key: зашифрованный симметричный ключ
        :param plaintext: исходный текст для шифрования
        :return: зашифрованные данные
        """

        symmetric_key = Asymmetrical.decrypt_by_private_key(
            private_key, encrypted_sym_key
        )#плохой комментарий
        return Symmetrical.encrypt_text(key=symmetric_key, text=plaintext)





    @staticmethod
    def decrypt_data(
            priv_key: RSAPrivateKey,
            enc_sym_key: bytes,
            encrypted_data: bytes) -> str:
        """
        Дешифрование данных с помощью гибридного шифрования
        :param priv_key: закрытый RSA ключ
        :param enc_sym_key: зашифрованный симметричный ключ
        :param encrypted_data: зашифрованные данные
        :return: расшифрованный текст
        """
        sk = Asymmetrical.decrypt_by_private_key(priv_key,enc_sym_key)
        return Symmetrical.decrypt_text(key=sk, encrypted_data=encrypted_data)