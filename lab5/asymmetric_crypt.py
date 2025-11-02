import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from loguru import logger


def read_file(file_path):
    """
    Reads file content in binary mode with error handling.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"file {file_path} not found")
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            logger.debug(f"File successfully read: {file_path}, size: {len(content)} bytes")
            return content
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading file {file_path}: {str(e)}")
        raise


def write_file(file_path, data):
    """
    Writes data to file in binary mode with error handling using match/case.
    """
    output_dir = os.path.dirname(file_path)
    try:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            f.write(data)
        logger.debug(f"File successfully written: {file_path}, size: {len(data)} bytes")
    except OSError as e:
        match str(e).lower().find("mkdir") >= 0:
            case True:
                logger.error(f"Error creating directory {output_dir}: {str(e)}")
            case False:
                logger.error(f"Error writing file {file_path}: {str(e)}")
        raise
    except IOError as e:
        logger.error(f"Error writing file {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An unknown error occurred while writing the file {file_path}: {str(e)}")
        raise


def generate_asymmetric_keys():
    """
    Generates RSA key pair.
    """
    logger.info("Generating RSA key pair...")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    logger.info("RSA key pair generated successfully")
    return private_key, public_key


def encrypt_symmetric_key(sym_key, public_key, path):
    """
    Encrypts symmetric key using RSA-OAEP.
    """
    logger.info("Encrypting symmetric key...")
    encrypted_key = public_key.encrypt(
        sym_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    write_file(path, encrypted_key)
    logger.info(f"Encrypted symmetric key saved to {path}")


def decrypt_symmetric_key(encrypted_key_path, private_key_path):
    """
    Decrypts symmetric key.
    """
    logger.info("Decrypting symmetric key...")
    encrypted_key = read_file(encrypted_key_path)
    private_key_data = read_file(private_key_path)
    private_key = load_pem_private_key(private_key_data, password=None)

    sym_key = private_key.decrypt(
        encrypted_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    logger.info("Symmetric key decrypted successfully")
    return sym_key