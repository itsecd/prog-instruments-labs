from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class Asymmetric:
    @staticmethod
    def generate_asymmetric_keys() -> tuple:

        """
        generates asymmetric keys
        :return: asymmetric keys
        """

        keys = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_key = keys
        public_key = keys.public_key()

        return private_key, public_key

    @staticmethod
    def encrypt_symmetric_key(public_key: rsa.RSAPublicKey, symmetric_key: bytes) -> bytes:

        """
        encrypts symmetric key with rsa
        :param public_key: public asymmetric key
        :param symmetric_key: symmetric key to encrypt
        :return: encrypted key
        """

        c_key = public_key.encrypt(symmetric_key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))

        return c_key

    @staticmethod
    def decrypt_symmetric_key(private_key: rsa.RSAPrivateKey, c_key: bytes) -> bytes:

        """
        decrypts symmetric key
        :param private_key: private asymmetric key
        :param c_key: encrypted symmetric key
        :return: decrypted symmetric key
        """

        dc_key = private_key.decrypt(c_key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))

        return dc_key
