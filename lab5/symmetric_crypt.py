import os
from loguru import logger

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def read_file(file_path):
    """
    Reads file content in binary mode with error handling.
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            logger.debug(f"File successfully read: {file_path}, size: {len(content)} bytes")
            return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File {file_path} not found.")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        raise PermissionError(f"File access denied {file_path}.")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise IOError(f"Error reading file {file_path}: {e}")


def write_file(file_path, data):
    """
    Writes data to file in binary mode with error handling.
    """
    try:
        with open(file_path, 'wb') as f:
            f.write(data)
        logger.debug(f"File successfully written: {file_path}, size: {len(data)} bytes")
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        raise PermissionError(f"File access denied {file_path}.")
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        raise IOError(f"Error writing file {file_path}: {e}")


def generate_symmetric_key():
    """
    Generates 128-bit key for SEED.
    """
    logger.info("Generating symmetric SEED key...")
    key = os.urandom(16)
    logger.debug(f"Symmetric key generated, size: {len(key)} bytes")
    return key


def encrypt_file(input_path, output_path, sym_key):
    """
    Encrypts file using SEED.
    """
    logger.info(f"Encrypting file {input_path}...")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file {input_path} not found")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.debug(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    iv = os.urandom(16)
    cipher = Cipher(algorithms.SEED(sym_key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    plaintext = read_file(input_path)
    logger.debug(f"Plaintext read, size: {len(plaintext)} bytes")

    padder = padding.ANSIX923(128).padder()
    padded_text = padder.update(plaintext) + padder.finalize()
    logger.debug(f"Plaintext padded, size: {len(padded_text)} bytes")

    ciphertext = encryptor.update(padded_text) + encryptor.finalize()
    logger.debug(f"Ciphertext generated, size: {len(ciphertext)} bytes")

    try:
        write_file(output_path, iv + ciphertext)
        logger.success(f"Encrypted file saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving encrypted file: {e}")
        raise


def decrypt_file(input_path, output_path, sym_key):
    """
    Decrypts file using SEED.
    """
    logger.info(f"Decrypting file {input_path}...")
    if not os.path.exists(input_path):
        logger.error(f"Encrypted file not found: {input_path}")
        raise FileNotFoundError(f"Encrypted file {input_path} not found")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logger.debug(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    data = read_file(input_path)
    logger.debug(f"Encrypted data read, size: {len(data)} bytes")
    
    iv, ciphertext = data[:16], data[16:]
    logger.debug(f"IV extracted: {len(iv)} bytes, ciphertext: {len(ciphertext)} bytes")

    cipher = Cipher(algorithms.SEED(sym_key), modes.CBC(iv))
    decryptor = cipher.decryptor()

    padded_text = decryptor.update(ciphertext) + decryptor.finalize()
    logger.debug(f"Decrypted padded text, size: {len(padded_text)} bytes")

    unpadder = padding.ANSIX923(128).unpadder()
    plaintext = unpadder.update(padded_text) + unpadder.finalize()
    logger.debug(f"Plaintext unpadded, size: {len(plaintext)} bytes")

    try:
        write_file(output_path, plaintext)
        logger.success(f"Decrypted file saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving decrypted file: {e}")
        raise