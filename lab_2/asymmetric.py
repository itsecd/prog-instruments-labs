from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from serialize import serialization


class AsymmetricCipher:
    @staticmethod
    def generate_keys():
        """
        Generates RSA private and public key pair

        Returns: (private_key, public_key) RSA key pair
        """
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            return private_key, private_key.public_key()
        except Exception as e:
            raise Exception(f"Error at generate asymmetric key: {str(e)}")

    @staticmethod
    def encrypt(public_key, data):
        """
        Encrypts data using RSA public key with OAEP padding
    
        :param public_key: RSA public key for encryption
        :param data: Data to encrypt
        :return: Encrypted data
        """
        try:
            return public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        except Exception as e:
            raise Exception(f"Error at encrypting asymmetric key: {str(e)}")

    @staticmethod
    def decrypt(private_key, encrypted_data):
        """
        Decrypts data using RSA private key with OAEP padding

        :param private_key: RSA private key for decryption
        :param encrypted_data: Data to decrypt
        :return: Decrypted data
        """
        try:
            return private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        except Exception as e:
            raise Exception(f"Error at decrypting asymmetric key: {str(e)}")

    @staticmethod
    def load_private_key(key_data):
        """
        Loads RSA private key from PEM encoded data

        :param key_data: PEM encoded private key
        :return: Loaded private key
        """
        return serialization.load_pem_private_key(
            key_data,
            password=None,
            backend=default_backend()
        )
