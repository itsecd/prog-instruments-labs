from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend


class Asymmetric:
    @staticmethod
    def asymmetrical_keygen() -> tuple:
        """
        Генерирует пару RSA-ключей (приватный и публичный)
        :return: Кортеж из приватного и публичного RSA-ключей
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def symmetrical_key_encryptor(
        symmetric_key: bytes, public_key: rsa.RSAPublicKey
    ) -> bytes:
        """
        Шифрует симметричный ключ с помощью публичного RSA-ключа
        :param symmetric_key: Симметричный ключ для шифрования
        :param public_key: Публичный RSA-ключ
        :return: Зашифрованный симметричный ключ
        """
        ciphertext = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return ciphertext

    @staticmethod
    def symmetrical_key_decryptor(
        ciphertext: bytes, private_key: rsa.RSAPrivateKey
    ) -> bytes:
        """
        Расшифровывает симметричный ключ с помощью приватного RSA-ключа
        :param ciphertext: Зашифрованный симметричный ключ
        :param private_key: Приватный RSA-ключ
        :return: Расшифрованный симметричный ключ
        """
        symmetric_key = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return symmetric_key
