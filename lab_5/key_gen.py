import logging
import os

from RSA import (
    encrypt_data,
    generate_rsa_keys,
    serialize_public_key,
    serialize_private_key,
)

logger = logging.getLogger(__name__)


def key_gen(
    lenght: int,
    public_key_pem_path: str,
    private_key_pem_path: str,
    symmetric_key_path: str,
):
    """
    Generate RSA pair of keys and symmetric key.
    :param lenght: Count bits of symmetric key
    :param public_key_pem_path: Path for save public key
    :param private_key_pem_path: Path for save private key
    :param symmetric_key_path: Path for save symmetric key
    """
    logger.info(
        "Key generation started: symmetric_key_bits=%s, public_key=%s, "
        "private_key=%s, symmetric_key=%s",
        lenght,
        public_key_pem_path,
        private_key_pem_path,
        symmetric_key_path,
    )
    try:
        private_key, public_key = generate_rsa_keys()
        serialize_public_key(public_key, public_key_pem_path)
        serialize_private_key(private_key, private_key_pem_path)
        key = os.urandom(lenght // 8)
        serialize_symmetric_key(encrypt_data(key, public_key), symmetric_key_path)
        logger.info("Key generation completed successfully")
    except Exception:
        logger.exception("Key generation failed")
        raise


def serialize_symmetric_key(key, symmetric_key_path: str):
    """
    Save the symmetric key to file
    :param key: source symmetric key
    :param symmetric_key_path: Path for save symmetric key
    """
    logger.debug("Saving encrypted symmetric key to %s", symmetric_key_path)
    try:
        with open(symmetric_key_path, "wb") as key_file:
            key_file.write(key)
        logger.debug("Encrypted symmetric key saved: %s", symmetric_key_path)
    except Exception:
        logger.exception("Failed to save symmetric key to %s", symmetric_key_path)
        raise


def deserialize_symmetric_key(symmetric_key_path: str):
    """
    Read symmetric key.
    :param symmetric_key_path: Path for read symmetric key
    :return: symmetric key
    """
    logger.debug("Reading encrypted symmetric key from %s", symmetric_key_path)
    try:
        with open(symmetric_key_path, "rb") as key_file:
            data = key_file.read()
        logger.debug("Encrypted symmetric key read: %s bytes", len(data))
        return data
    except FileNotFoundError:
        logger.error("Symmetric key file not found: %s", symmetric_key_path)
        raise
    except Exception:
        logger.exception("Failed to read symmetric key from %s", symmetric_key_path)
        raise
