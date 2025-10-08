from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey


class AsymmetricCrypto:
    """Класс для работы с асимметричным шифрованием (RSA)"""

    @staticmethod
    def  generate_keys() -> tuple[RSAPrivateKey, RSAPublicKey]:
        """Генерирует пару RSA ключей"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        return private_key, private_key.public_key()

    @staticmethod
    def encrypt_with_public_key(public_key: RSAPublicKey, data: bytes) -> bytes:
        """Шифрует данные публичным ключом"""
        return public_key.encrypt(
            data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    @staticmethod
    def decrypt_with_private_key(private_key: RSAPrivateKey, encrypted_data: bytes) -> bytes:
        """Дешифрует данные приватным ключом"""
        return private_key.decrypt(
            encrypted_data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )