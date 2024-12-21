from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from logger_config import setup_logger

logger = setup_logger(__name__)


class Asymmetric:
    """
    A class that implements asymmetric encryption and decryption using the RSA algorithm.

    Attributes
        private_key: The private key.
        public_key: The public key.
    """

    def __init__(self):
        self.private_key = None
        self.public_key = None
        logger.info("Asymmetric instance created.")

    def generate_keys(self) -> None:
        """
        Generates a new RSA private and public key pair.
        """
        logger.info("Generating RSA key pair.")
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()
        logger.info("RSA key pair generated.")

    def serialization_public(self, public_path: str) -> None:
        """
        Serializes the RSA public key to files.

        Parameters
            public_path: The path to the file where the public key will be saved.
        """
        logger.info(f"Serializing public key to: {public_path}")
        try:
            with open(public_path, 'wb') as public_out:
                public_out.write(self.public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                              format=serialization.PublicFormat.SubjectPublicKeyInfo))
            logger.info(f"Public key successfully written to file: {public_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {public_path}")
            print(f"The file '{public_path}' was not found.")
        except Exception as e:
            logger.error(f"Error during public key serialization: {str(e)}")
            print(f"Error: {str(e)}")

    def serialization_private(self, private_path: str) -> None:
        """
        Serializes the RSA private key to files.

        Parameters
            private_path: The path to the file where the private key will be saved.
        """
        logger.info(f"Serializing private key to: {private_path}")
        try:
            with open(private_path, 'wb') as private_out:
                private_out.write(self.private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                                 format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                                 encryption_algorithm=serialization.NoEncryption()))
            logger.info(f"Private key successfully written to file: {private_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {private_path}")
            print(f"The file '{private_path}' was not found.")
        except Exception as e:
            logger.error(f"Error during private key serialization: {str(e)}")
            print(f"Error: {str(e)}")

    def public_key_deserialization(self, public_path: str) -> None:
        """
        Deserializes the RSA public key from a file.

        Parameters
            public_path: The path to the file containing the public key.
        """
        logger.info(f"Deserializing public key from: {public_path}")
        try:
            with open(public_path, 'rb') as pem_in:
                public_bytes = pem_in.read()
            self.public_key = load_pem_public_key(public_bytes)
            logger.info(f"Public key successfully deserialized from: {public_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {public_path}")
            print(f"The file '{public_path}' was not found.")
        except Exception as e:
            logger.error(f"Error during public key deserialization: {str(e)}")
            print(f"Error: {str(e)}")

    def private_key_deserialization(self, private_path: str) -> None:
        """
        Deserializes the RSA private key from a file.

        Parameters
            private_path: The path to the file containing the private key.
        """
        logger.info(f"Deserializing private key from: {private_path}")
        try:
            with open(private_path, 'rb') as pem_in:
                private_bytes = pem_in.read()
            self.private_key = load_pem_private_key(private_bytes, password=None)
            logger.info(f"Private key successfully deserialized from: {private_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {private_path}")
            print(f"The file '{private_path}' was not found.")
        except Exception as e:
            logger.error(f"Error during private key deserialization: {str(e)}")
            print(f"Error: {str(e)}")

    def encrypt(self, symmetric_key: bytes) -> bytes:
        """
        Encrypts a symmetric key using the public key.

        Parameters
            symmetric_key (bytes): The symmetric key to be encrypted.
        Returns
            The encrypted symmetric key.
        """
        logger.info("Encrypting symmetric key using public key.")
        encrypted_symmetric_key = self.public_key.encrypt(symmetric_key,
                                                          padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                       algorithm=hashes.SHA256(), label=None))
        logger.info("Symmetric key encrypted.")
        return encrypted_symmetric_key

    def decrypt(self, symmetric_key: bytes) -> bytes:
        """
        Decrypts a symmetric key using the private key.

        Parameters
            symmetric_key (bytes): The encrypted symmetric key to be decrypted.
        Returns
            The decrypted symmetric key.
        """
        logger.info("Decrypting symmetric key using private key.")
        if self.private_key is None:
            logger.error("Private key has not been initialized.")
            raise ValueError("Private key has not been initialized.")
        decrypted_symmetric_key = self.private_key.decrypt(symmetric_key,
                                                           padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                        algorithm=hashes.SHA256(), label=None))
        logger.info("Symmetric key decrypted.")
        return decrypted_symmetric_key
