import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

def symmetric_key_generation(key_size: int = 32) -> bytes:
    """
    Generate symmetric keys

    Returns:
        bytes: Random 32-byte symmetric key
    """
    return os.urandom(key_size)


def symmetric_key_writter(filepath: str, symmetric_key: bytes):
    """
    Write symmetric key

    Args:
        filepath: Path to output file
        symmetric_key: Symmetric key bytes to write
    """
    with open(filepath, "wb") as file:
        file.write(symmetric_key)


def text_encryption(text: str, key: bytes, nonce_size: int) -> tuple:
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
    
    nonce = os.urandom(nonce_size)
    counter = 0
    full_nonce = counter.to_bytes(8, 'little') + nonce

    cipher = Cipher(algorithms.ChaCha20(key, full_nonce), mode=None)
    encryptor = cipher.encryptor()
    return encryptor.update(text.encode('utf-8')), nonce


def text_decryption(c_text: bytes, key: bytes, key_size: int, nonce_size: int) -> str:
    """
    Encryption text with ChaCha20

    Args:
        c_text: Encrypted bytes where first nonce_size bytes are nonce
        key: key_size-byte symmetric key for decryption
    
    Returns:
        str: Decrypted plain text
    
    Raises:
        ValueError: If key length is not key_size bytes or encrypted text is too short
    """
    if len(key) != key_size:
        raise ValueError(f"Key length must be 32. Your key length - {len(key)}")
    
    
    if len(c_text) < nonce_size:
        raise ValueError(f"Encrypted text length must higher than nonce_size. Your text length - {len(key)}")
    
    nonce = c_text[:nonce_size]
    c_text = c_text[nonce_size:]
    counter = 0
    full_nonce = counter.to_bytes(nonce_size, 'little') + nonce

    cipher = Cipher(algorithms.ChaCha20(key, full_nonce), mode=None)
    decryptor = cipher.decryptor()
    return decryptor.update(c_text).decode('utf-8')