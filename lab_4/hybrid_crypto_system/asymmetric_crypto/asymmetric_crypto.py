from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asymmetric_padding


class AsymmetricCrypto:
    @staticmethod
    def generate_keys(key_length: int) -> tuple:
        """
        Asymmetric key generation method
        :param key_length: key size
        :return: private_key, public_key
        """

        keys = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_key = keys
        public_key = keys.public_key()

        return private_key, public_key

    @staticmethod
    def encrypt_symmetric_key(symmetric_key, public_key):
        """
        Symmetric key encryption method
        :param symmetric_key: the key to encrypt
        :param public_key: public asymmetric key
        :return: encrypted symmetric key
        """
        return public_key.encrypt(
            symmetric_key,
            asymmetric_padding.OAEP(
                mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    @staticmethod
    def decrypt_symmetric_key(encrypted_symmetric_key, private_key):
        """
        Symmetric key decryption method
        :param encrypted_symmetric_key: the key to decrypt
        :param private_key: private asymmetric key
        :return: decrypted symmetric key
        """
        return private_key.decrypt(
            encrypted_symmetric_key,
            asymmetric_padding.OAEP(
                mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
