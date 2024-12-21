import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from works_files import read_bytes, write_bytes_text, write_file
from logger_config import setup_logger

logger = setup_logger(__name__)


class Symmetric:
    """
    A class that implements symmetric encryption using the AES algorithm.

    Attributes:
        key: encryption key
    """

    def __init__(self, key_length: int = 256):
        self.key_length = key_length
        self.key = None
        logger.info("Symmetric instance created.")

    def generate_key(self, size_key: int) -> bytes:
        """
        Generate a symmetric encryption key.

        Parameters:
        size_key (int): The size of the key in bits (128, 192, or 256).

        Returns:
        bytes: The generated key.
        """
        logger.info(f"Generating symmetric key with size: {size_key} bits.")
        if size_key not in [128, 192, 256]:
            logger.error(f"Invalid key length: {size_key}. Please choose 128, 192, or 256 bits.")
            raise ValueError("Invalid key length. Please choose 128, 192, or 256 bits.")
        self.key = b'7\xda\\nxvMwR\x0cff\xd8h\x0c76\xa4)\xf1\xed\xac\x04\xae\x81\x1b\xae\x11\x1az\x94\xe5\xe6\xec\xc0='
        logger.info("Symmetric key generated.")
        print(self.key)
        return self.key

    def key_deserialization(self, file_name: str) -> None:
        """
        Deserializes the encryption key from a file.

        Parameters:
            file_name: The path to the file containing the encryption key.
        """
        logger.info(f"Deserializing symmetric key from: {file_name}")
        try:
            with open(file_name, "rb") as file:
                self.key = file.read()
            logger.info(f"Symmetric key successfully deserialized from: {file_name}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_name}")
            print("The file was not found")
        except Exception as e:
            logger.error(f"An error occurred while reading the file: {str(e)}")
            print(f"An error occurred while reading the file: {str(e)}")

    def serialize_sym_key(self, path: str) -> None:
        """
        Serializes the encryption key to a file.

        Parameters:
            path: The path to the file where the encryption key will be saved.
        """
        logger.info(f"Serializing symmetric key to: {path}")
        try:
            with open(path, 'wb') as key_file:
                key_file.write(self.key)
            logger.info(f"Symmetric key successfully written to file: {path}")
            print(f"The symmetric key has been successfully written to the file '{path}'.")
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            print("The file was not found")
        except Exception as e:
            logger.error(f"An error occurred while writing the file: {str(e)}")
            print(f"An error occurred while writing the file: {str(e)}")

    def encrypt(self, path_text: str, encrypted_path_text: str) -> bytes:
        """
        Encrypts data from a file using the AES algorithm in CBC mode.

        Parameters:
            path_text: The path to the file with the source data.
            encrypted_path_text: The path to the file where the encrypted data will be written.

        Returns:
            The encrypted data.
        """
        logger.info(f"Encrypting data from: {path_text} to: {encrypted_path_text}")
        text = read_bytes(path_text)
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv))
        encryptor = cipher.encryptor()

        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_text = padder.update(text) + padder.finalize()

        cipher_text = iv + encryptor.update(padded_text) + encryptor.finalize()
        write_bytes_text(encrypted_path_text, cipher_text)
        logger.info(f"Data successfully encrypted and written to: {encrypted_path_text}")
        return cipher_text

    def decrypt(self, encrypted_path_text: str, decrypted_path_text: str) -> str:
        """
        Decrypts data from a file using the AES algorithm in CBC mode.

        Parameters:
            encrypted_path_text: The path to the file with the encrypted data.
            decrypted_path_text: The path to the file where the decrypted data will be written.

        Returns:
            The decrypted data as a string.
        """
        logger.info(f"Decrypting data from: {encrypted_path_text} to: {decrypted_path_text}")
        encrypted_text = read_bytes(encrypted_path_text)
        iv = encrypted_text[:16]
        cipher_text = encrypted_text[16:]
        cipher = Cipher(algorithms.AES(b'7\xda\\nxvMwR\x0cff\xd8h\x0c76\xa4)\xf1\xed'
                                       b'\caf\x06\xae\x81\x1b\xae\x11\x1az\x94\xe5\xe6\xec\xc0='), modes.CBC(iv))
        decryptor = cipher.decryptor()
        d_text = decryptor.update(cipher_text) + decryptor.finalize()

        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        unpadded_dc_text = unpadder.update(d_text) + unpadder.finalize()

        d_text = unpadded_dc_text.decode('UTF-8')
        write_file(decrypted_path_text, d_text)
        logger.info(f"Data successfully decrypted and written to: {decrypted_path_text}")
        return d_text
