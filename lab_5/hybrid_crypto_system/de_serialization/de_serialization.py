from typing import Union

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from filehandler import FileHandler
from hybrid_crypto_system.de_serialization.constants import KeyTypes
from hybrid_crypto_system.logger.logger_config import logger


class DeSerialization:
    """Serialization and deserialization operations"""

    @staticmethod
    def serialization_rsa_key(
            key: Union[rsa.RSAPrivateKey, rsa.RSAPublicKey],
            key_type: KeyTypes
    ) -> bytes:
        """
        Serialization rsa asymmetric key
        :param key_dir: directory to save the key
        :param key: key to save
        :param key_type: type of key - private or public
        :return: serialized key
        """
        logger.info("Starting asymmetric key serialization - key type: %s", key_type)
        try:
            serialized_key = None
            match key_type:
                case KeyTypes.private:
                    serialized_key = key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                    logger.debug("Serializing private key")
                case KeyTypes.public:
                    serialized_key =  key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                    logger.debug("Serializing public key")
            logger.info("Asymmetric key serialized successfully")
            logger.debug("Serialized key length: %d bytes", len(serialized_key))
            return serialized_key
        except Exception as e:
            logger.error("Failed to serialize key: %s", str(e))
            raise

    @staticmethod
    def deserialization_rsa_key(
            key: bytes,
            key_type: KeyTypes
    ) -> Union[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Deserialization rsa asymmetric key
        :param key: the key to deserialize
        :param key_type: type of key - private or public
        :return: deserialized key
        """
        logger.info("Starting asymmetric key deserialization - key type: %s", key_type)
        logger.debug("Input key data length: %d bytes", len(key))
        try:
            deserialized_key = None
            match key_type:
                case KeyTypes.private:
                    deserialized_key =  serialization.load_pem_private_key(key, password=None)
                    logger.debug("Private key deserialized")
                case KeyTypes.public:
                    deserialized_key = serialization.load_pem_public_key(key)
                    logger.debug("Public key deserialized")

            logger.info("Asymmetric key deserialized successfully - key type: %s", key_type)
            return deserialized_key
        except Exception as e:
            logger.error("Failed to deserialize key: %s", str(e))
            raise

    @staticmethod
    def serialization_data(
            data_dir: str,
            data: bytes
    ) -> None:
        """
        Method serializes the data and saves it to a file
        :param data_dir: directory to save file
        :param data: data to serialize
        :return: None
        """
        logger.info("Starting data serialization")
        try:
            FileHandler.save_data(data_dir, data, "wb")
            logger.debug("File operation completed - write binary mode")
        except FileNotFoundError:
            logger.error("File not found during serialization: %s", data_dir)
            raise
        except Exception as e:
            logger.error("Failed to serialize data: %s", str(e))
            raise


    @staticmethod
    def deserialization_data(
            data_dir: str
    ) -> bytes:
        """
        Method deserializes the data from the file
        :param data_dir: directory to serialized file
        :return: deserialized data
        """
        logger.info("Starting data deserialization")
        try:
            data = FileHandler.read_data(data_dir, "rb")
            logger.info("Data deserialization successful")
            logger.debug("File operation completed - read binary mode")
            return data
        except FileNotFoundError:
            logger.error("File not found during deserialization: %s", data_dir)
            raise
        except Exception as e:
            logger.error("Failed to deserialize data: %s", str(e))
            raise
