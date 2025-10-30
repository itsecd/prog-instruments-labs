from typing import Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asymmetric_padding

from hybrid_crypto_system.asymmetric_crypto.constants import KEY_SIZE, PUBLIC_EXPONENT
from hybrid_crypto_system.logger.logger_config import logger


class AsymmetricCrypto:
    @staticmethod
    def generate_keys() -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Asymmetric keys generation method
        :return: private_key, public_key
        """
        logger.info("Starting asymmetric key generation")
        try:
            keys = rsa.generate_private_key(public_exponent=PUBLIC_EXPONENT, key_size=KEY_SIZE)
            private_key = keys
            public_key = keys.public_key()
            logger.info(f"Asymmetric keys generated successfully")
            return private_key, public_key
        except Exception as e:
            logger.error("Failed to generate asymmetric keys: %s", str(e))
            raise

    @staticmethod
    def encrypt_symmetric_key(
            symmetric_key: bytes,
            public_key: rsa.RSAPublicKey
    ) -> bytes:
        """
        Symmetric key encryption method
        :param symmetric_key: the key to encrypt
        :param public_key: public asymmetric key
        :return: encrypted symmetric key
        """
        logger.info("Starting symmetric key encryption")
        try:
            encrypted_key = public_key.encrypt(
                symmetric_key,
                asymmetric_padding.OAEP(
                    mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            logger.info("Symmetric key encrypted successfully")
            return encrypted_key
        except Exception as e:
            logger.error("Failed to encrypt symmetric key: %s", str(e))
            raise

    @staticmethod
    def decrypt_symmetric_key(
            encrypted_symmetric_key: bytes,
            private_key: rsa.RSAPrivateKey
    ) -> bytes:
        """
        Symmetric key decryption method
        :param encrypted_symmetric_key: the key to decrypt
        :param private_key: private asymmetric key
        :return: decrypted symmetric key
        """
        logger.info("Starting symmetric key decryption")
        try:
            decrypted_key = private_key.decrypt(
                encrypted_symmetric_key,
                asymmetric_padding.OAEP(
                    mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            logger.info("Symmetric key decrypted successfully")
            return decrypted_key
        except Exception as e:
            logger.error("Failed to decrypt symmetric key: %s", str(e))
            raise
