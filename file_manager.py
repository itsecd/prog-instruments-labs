from typing import Union

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key


class FileManager:
    """Класс для работы с файлами криптографических ключей и данных"""

    @staticmethod
    def save_file(path: str, data: bytes) -> None:
        """Сохраняет данные в файл"""
        with open(path, 'wb') as f:
            f.write(data)

    @staticmethod
    def load_file(path: str) -> bytes:
        """Загружает данные из файла"""
        with open(path, 'rb') as f:
            return f.read()

    @staticmethod
    def save_key(key: Union[RSAPrivateKey, RSAPublicKey], path: str) -> None:
        """Сохраняет криптографический ключ в файл"""
        with open(path, 'wb') as f:
            if isinstance(key, RSAPrivateKey):
                private_bytes = key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                )
                f.write(private_bytes)
            else:
                public_bytes = key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                f.write(public_bytes)

    @staticmethod
    def load_private_key(path: str) -> RSAPrivateKey:
        """Загружает приватный RSA ключ из файла"""
        with open(path, 'rb') as f:
            return load_pem_private_key(f.read(), password=None)

    @staticmethod
    def load_public_key(path: str) -> RSAPublicKey:
        """Загружает публичный RSA ключ из файла"""
        with open(path, 'rb') as f:
            return load_pem_public_key(f.read())