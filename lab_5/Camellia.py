import logging
import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from key_gen import deserialize_symmetric_key
from RSA import deserialize_private_key, decrypt_data
from text_processor import save_text, read_text, read_chipher

logger = logging.getLogger(__name__)


def encrypt(
    text_file_path: str,
    symmetric_key_path: str,
    private_key_pem_path: str,
    encrypted_file_path: str,
):
    """
    Encrypt data with symmetric encryption using RSA key encryption
    :param text_file_path: Path to source file
    :param symmetric_key_path: Path to serialized encrypted symmetric key
    :param private_key_pem_path: Path to serialized private RSA key
    :param encrypted_file_path: Path for save encrypted data
    """
    logger.info(
        "Encryption started: input=%s, output=%s",
        text_file_path,
        encrypted_file_path,
    )
    try:
        text = read_text(text_file_path)
        logger.debug("Source file size: %s bytes", len(text))

        iv = os.urandom(16)
        logger.debug("Generated IV for Camellia-CBC: %s bytes", len(iv))

        key = decrypt_data(
            deserialize_symmetric_key(symmetric_key_path),
            deserialize_private_key(private_key_pem_path),
        )
        logger.debug("Symmetric key decrypted with RSA private key: %s bytes", len(key))

        cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
        encryptor = cipher.encryptor()

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(text) + padder.finalize()

        c_text = encryptor.update(padded_data) + encryptor.finalize()
        logger.debug(
            "Camellia encryption completed: ciphertext_size=%s bytes", len(c_text)
        )

        save_text(iv + c_text, encrypted_file_path)
        logger.info("Encryption completed successfully: %s", encrypted_file_path)
    except Exception:
        logger.exception(
            "Encryption failed: input=%s, output=%s",
            text_file_path,
            encrypted_file_path,
        )
        raise


def decrypt(
    text_file_path: str,
    symmetric_key_path: str,
    private_key_pem_path: str,
    decrypted_file_path: str,
):
    """
    Decrypt data with symmetric encryption using RSA key encryption
    :param text_file_path: Path to encrypted file
    :param symmetric_key_path: Path to serialized encrypted symmetric key
    :param private_key_pem_path: Path to serialized private RSA key
    :param decrypted_file_path: Path for save decrypted data
    """
    logger.info(
        "Decryption started: input=%s, output=%s",
        text_file_path,
        decrypted_file_path,
    )
    try:
        iv, c_text = read_chipher(text_file_path)
        logger.debug("Ciphertext size without IV: %s bytes", len(c_text))

        key = decrypt_data(
            deserialize_symmetric_key(symmetric_key_path),
            deserialize_private_key(private_key_pem_path),
        )
        logger.debug("Symmetric key decrypted with RSA private key: %s bytes", len(key))

        cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        dc_text = decryptor.update(c_text) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        dc_text = unpadder.update(dc_text) + unpadder.finalize()
        logger.debug(
            "Camellia decryption completed: plaintext_size=%s bytes", len(dc_text)
        )

        save_text(dc_text, decrypted_file_path)
        logger.info("Decryption completed successfully: %s", decrypted_file_path)
    except Exception:
        logger.exception(
            "Decryption failed: input=%s, output=%s",
            text_file_path,
            decrypted_file_path,
        )
        raise
