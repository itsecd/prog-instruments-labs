import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

def symmetric_key_generation()->bytes:
    """
    Generate symmetric keys

    Returns:
        bytes: Random 32-byte symmetric key
    """
    key = os.urandom(32)
    return key


def symmetric_key_writter(filepath: str, symmetric_key: bytes):
    """
    Write symmetric key

    Args:
        filepath: Path to output file
        symmetric_key: Symmetric key bytes to write
    """
    with open(filepath, "wb") as file:
        file.write(symmetric_key)


def text_encryption(text: str, key: bytes) -> tuple:
    """
    Encryption text with ChaCha20

    Args:
        text: Plain text string to encrypt
        key: 32-byte symmetric key for encryption
    
    Returns:
        tuple: (encrypted_text_bytes, nonce_bytes)
    
    Raises:
        ValueError: If key length is not 32 bytes
    """
    if len(key) != 32:
        raise ValueError(f"Key length must be 32. Your key length - {len(key)}")
    
    nonce = os.urandom(8)
    counter = 0
    full_nonce = counter.to_bytes(8, 'little') + nonce

    cipher = Cipher(algorithms.ChaCha20(key, full_nonce), mode=None)
    encryptor = cipher.encryptor()
    return encryptor.update(text.encode('utf-8')), nonce


def text_decryption(c_text: bytes, key: bytes) -> str:
    """
    Encryption text with ChaCha20

    Args:
        c_text: Encrypted bytes where first 8 bytes are nonce
        key: 32-byte symmetric key for decryption
    
    Returns:
        str: Decrypted plain text
    
    Raises:
        ValueError: If key length is not 32 bytes or encrypted text is too short
    """
    if len(key) != 32:
        raise ValueError(f"Key length must be 32. Your key length - {len(key)}")
    
    
    if len(c_text) < 8:
        raise ValueError(f"Encrypted text length must higher than 8. Your text length - {len(key)}")
    
    nonce = c_text[:8]
    c_text = c_text[8:]
    counter = 0
    full_nonce = counter.to_bytes(8, 'little') + nonce

    cipher = Cipher(algorithms.ChaCha20(key, full_nonce), mode=None)
    decryptor = cipher.decryptor()
    return decryptor.update(c_text).decode('utf-8')