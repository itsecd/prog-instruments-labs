import os
from typing import Tuple

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, modes

from hybrid_crypto_system.asymmetric_crypto.asymmetric_crypto import AsymmetricCrypto
from hybrid_crypto_system.de_serialization.de_serialization import DeSerialization
from hybrid_crypto_system.de_serialization.constants import KeyTypes
from hybrid_crypto_system.symmetric_crypto.constants import BYTES
from hybrid_crypto_system.logger.logger_config import logger


class SymmetricCrypto:

    @staticmethod
    def generate_key(key_length: int) -> bytes:
        """
        Method to generate symmetric key
        :param key_length: key size
        :return: symmetric key
        """
        logger.info(f'Generating symmetric key with length: {key_length} bits')
        try:
            key = os.urandom(key_length // BYTES)
            logger.debug(f'Symmetric key generated successfully, length: {len(key)} bytes')
            return key
        except Exception as e:
            logger.error(f"Failed to generate symmetric key: {str(e)}")
            raise

    @staticmethod
    def get_iv() -> bytes:
        """
        Method to get iv
        :return: iv
        """
        return os.urandom(BYTES)

    @staticmethod
    def padding(block_size: int, text: bytes) -> bytes:
        """
        Method to padding data
        :param block_size: size of block
        :param text: text to padding
        :return: padded text
        """
        padder = padding.ANSIX923(block_size).padder()
        padded_text = padder.update(text) + padder.finalize()
        logger.debug(f'Padding completed - padded text length: {len(padded_text)} bytes')
        return padded_text


    @staticmethod
    def unpadding(block_size: int, text: bytes) -> bytes:
        """
        Method to unpadding data
        :param block_size: size of block
        :param text: text to unpadding
        :return: unpadded text
        """
        unpadder = padding.ANSIX923(block_size).unpadder()
        unpadded_text = unpadder.update(text) + unpadder.finalize()
        logger.debug(f'Unpadding completed - unpadded text length: {len(unpadded_text)} bytes')
        return unpadded_text

    @staticmethod
    def split_from_iv(text: bytes) -> Tuple[bytes, bytes]:
        """
        Method splits data and iv
        :param text: text to split
        :return: iv, split text
        """
        iv = text[-BYTES:]
        split_text = text[:-BYTES]
        return iv, split_text

    @staticmethod
    def encrypt_with_algorithm(
            algorithm,
            key: bytes,
            iv: bytes,
            padded_text: bytes
    ) -> bytes:
        """
        Method encrypts data with cipher algorithm
        :param algorithm: cipher algorithm
        :param key: key for encryption
        :param iv: iv for encryption
        :param padded_text: padded text
        :return: encrypted text
        """
        logger.info(f'Encrypting data with {algorithm.__name__}')
        try:
            cipher = Cipher(algorithm(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            encrypted_text = encryptor.update(padded_text) + encryptor.finalize()

            logger.info('Data encrypted successfully')
            logger.debug(f'Encrypted text length: {len(encrypted_text)} bytes')
            return encrypted_text
        except Exception as e:
            logger.error(f"Failed to encrypt data with algorithm {algorithm.__name__}: {str(e)}")
            raise

    @staticmethod
    def decrypt_with_algorithm(
            algorithm,
            key: bytes,
            iv: bytes,
            text: bytes
    ) -> bytes:
        """
        Method decrypts data with cipher algorithm
        :param algorithm: cipher algorithm
        :param key: key for decryption
        :param iv: iv for decryption
        :param text: encrypted text
        :return: decrypted padded text
        """
        logger.info(f'Decrypting data with {algorithm.__name__}')
        try:
            cipher = Cipher(algorithm(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            decrypted_text = decryptor.update(text) + decryptor.finalize()

            logger.info('Data decrypted successfully')
            logger.debug(f'Decrypted padded text length: {len(decrypted_text)} bytes')
            return decrypted_text
        except Exception as e:
            logger.error(f"Failed to decrypt data with algorithm {algorithm.__name__}: {str(e)}")
            raise

    @staticmethod
    def encrypt_data(
            cipher_algorithm,
            plain_text: bytes,
            private_bytes: bytes,
            encrypted_symmetric_key: bytes
    ) -> bytes:
        """
        Method to encrypt data from plain_text
        :param cipher_algorithm: cipher algorithm
        :param plain_text: text to encrypt
        :param private_bytes: private key to decrypt key
        :param encrypted_symmetric_key: key to encrypt data
        :return: encrypted data
        """
        logger.info('Starting data encryption')
        try:
            private_key = DeSerialization.deserialization_rsa_key(
                private_bytes,
                KeyTypes.private
            )

            symmetric_key = AsymmetricCrypto.decrypt_symmetric_key(
                encrypted_symmetric_key, private_key
            )

            iv = SymmetricCrypto.get_iv()
            padded_text = SymmetricCrypto.padding(
                cipher_algorithm.block_size,
                plain_text
            )

            encrypted_text = SymmetricCrypto.encrypt_with_algorithm(
                cipher_algorithm,
                symmetric_key,
                iv,
                padded_text
            )

            result = encrypted_text + iv
            logger.info('Symmetric data encryption completed successfully')
            logger.debug(f'Final encrypted data length: {len(result)} bytes')
            return result
        except Exception as e:
            logger.error(f"Failed to encrypt data: {str(e)}")
            raise

    @staticmethod
    def decrypt_data(
            cipher_algorithm,
            encrypted_data: bytes,
            private_bytes: bytes,
            encrypted_symmetric_key: bytes
    ) -> bytes:
        """
        Method to decrypt data from encrypted data
        :param cipher_algorithm: cipher algorithm
        :param encrypted_data: encrypted text
        :param private_bytes: private key as bytes
        :param encrypted_symmetric_key: symmetric key as bytes
        :return: decrypted data
        """
        logger.info('Starting symmetric data decryption process')
        try:
            private_key = DeSerialization.deserialization_rsa_key(private_bytes, KeyTypes.private)

            symmetric_key = AsymmetricCrypto.decrypt_symmetric_key(
                encrypted_symmetric_key, private_key
            )

            iv, text = SymmetricCrypto.split_from_iv(encrypted_data)

            decrypted_padded_text = SymmetricCrypto.decrypt_with_algorithm(
                cipher_algorithm,
                symmetric_key,
                iv,
                text
            )

            decrypted_text = SymmetricCrypto.unpadding(
                cipher_algorithm.block_size,
                decrypted_padded_text
            )

            logger.info('Symmetric data decryption completed successfully')
            logger.debug(f'Decrypted text length: {len(decrypted_text)} bytes')
            return decrypted_text
        except Exception as e:
            logger.error(f"Failed to decrypt data: {str(e)}")
            raise