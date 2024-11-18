import logging

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes


logger = logging.getLogger()
logger.setLevel("INFO")


class RSA:
    """
    A class providing methods for RSA generation key, encryption and decryption.
    """
    def __init__(self, private_key_path: str, public_key_path: str) -> None:
        """
        Initialize a new object of the class.
        Parameters:
        - private_key_path (str): Path to the private key.
        - public_key_path (str): Path to the public key.
        Returns:
        None
        """
        self.private_key = private_key_path
        self.public_key = public_key_path

    def generate_rsa_key(self) -> tuple:
        """
        Generates RSA private and public key pair.
        Returns:
             tuple: A tuple containing the private key and the corresponding public key.
        """
        keys = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        private_key = keys
        public_key = keys.public_key()
        logging.info("Asymmetric encryption keys have been generated.")
        return private_key, public_key

    def encrypt_rsa(self, public_key: rsa.RSAPublicKey, text: bytes) -> bytes:
        """
        Encrypts text using RSA public key.
        Parameters:
            public_key: The RSA public key used for encryption.
            text (bytes): The text to be encrypted.
        Returns:
            bytes: The encrypted text.
        """
        encrypt_text = public_key.encrypt(
            text,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        logging.info("The text is encrypted.")
        return encrypt_text

    def decrypt_rsa(self, private_key: rsa.RSAPrivateKey, text: str) -> bytes:
        """
        Decrypts text using RSA private key.
        Parameters:
            private_key (object): The RSA private key used for decryption.
            text (bytes): The text to be decrypted.
        Returns:
            bytes: The decrypted text.
        """
        decrypt_text = private_key.decrypt(
            text,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        logging.info("The text encrypted has been decrypted.")
        return decrypt_text