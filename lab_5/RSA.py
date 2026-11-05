import logging

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    load_pem_private_key,
)

logger = logging.getLogger(__name__)


def generate_rsa_keys():
    """
    Generate a pair of public and private keys.
    :return: Pair of public and private RSA keys
    """
    logger.debug("Generating RSA key pair: key_size=2048, public_exponent=65537")
    keys = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key = keys
    public_key = keys.public_key()
    logger.debug("RSA key pair generated")
    return private_key, public_key


def serialize_public_key(public_key, public_pem_path: str):
    """
    Save the key in the format .pem
    :param public_key: Source public key
    :param public_pem_path: Path for save public key
    """
    logger.debug("Saving public RSA key to %s", public_pem_path)
    try:
        with open(public_pem_path, "wb") as public_out:
            public_out.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
        logger.debug("Public RSA key saved: %s", public_pem_path)
    except Exception:
        logger.exception("Failed to save public RSA key to %s", public_pem_path)
        raise


def serialize_private_key(private_key, private_pem_path: str):
    """
    Save the key in the format .pem
    :param private_key: Source private key
    :param private_pem_path: Path for save private key
    """
    logger.debug("Saving private RSA key to %s", private_pem_path)
    try:
        with open(private_pem_path, "wb") as private_out:
            private_out.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
        logger.debug("Private RSA key saved: %s", private_pem_path)
    except Exception:
        logger.exception("Failed to save private RSA key to %s", private_pem_path)
        raise


def deserialize_public_key(public_pem_path: str):
    """
    Read the key in the format .pem.
    :param public_pem_path: Path for read public key
    :return: public RSA key
    """
    logger.debug("Reading public RSA key from %s", public_pem_path)
    try:
        with open(public_pem_path, "rb") as pem_in:
            public_bytes = pem_in.read()
        public_key = load_pem_public_key(public_bytes)
        logger.debug("Public RSA key loaded: %s", public_pem_path)
        return public_key
    except Exception:
        logger.exception("Failed to read public RSA key from %s", public_pem_path)
        raise


def deserialize_private_key(private_pem_path: str):
    """
    Read the key in the format .pem
    :param private_pem_path: Path for read private key
    :return: private RSA key
    """
    logger.debug("Reading private RSA key from %s", private_pem_path)
    try:
        with open(private_pem_path, "rb") as pem_in:
            private_bytes = pem_in.read()
        private_key = load_pem_private_key(
            private_bytes,
            password=None,
        )
        logger.debug("Private RSA key loaded: %s", private_pem_path)
        return private_key
    except Exception:
        logger.exception("Failed to read private RSA key from %s", private_pem_path)
        raise


def encrypt_data(text, public_key):
    """
    Encrypt data using an RSA public key
    :param text: Source data
    :param public_key: Source public key for data encrypting
    :return: Encrpyted data
    """
    logger.debug("RSA encryption started: plaintext_size=%s bytes", len(text))
    c_text = public_key.encrypt(
        text,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    logger.debug("RSA encryption completed: ciphertext_size=%s bytes", len(c_text))
    return c_text


def decrypt_data(text, private_key):
    """
    Decrypt data using an RSA private key.
    :param text: Encrypted data
    :param private_key: Source private key for data decrypting
    :return: Decrpyted data
    """
    logger.debug("RSA decryption started: ciphertext_size=%s bytes", len(text))
    dc_text = private_key.decrypt(
        text,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    logger.debug("RSA decryption completed: plaintext_size=%s bytes", len(dc_text))
    return dc_text
