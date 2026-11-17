"""Модуль симметричной криптографии (Camellia)"""

import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .utils import pad_data, unpad_data


def generate_camellia_key(crypto_config: dict) -> bytes:
    """Генерация случайного ключа для Camellia"""
    return os.urandom(crypto_config['symmetric_key_size'])


def encrypt_with_camellia(data: bytes, key: bytes, crypto_config: dict) -> bytes:
    """Шифрование данных алгоритмом Camellia в режиме CBC"""
    block_size = crypto_config['block_size']
    
    iv = os.urandom(block_size)
    cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    
    padded_data = pad_data(data, block_size)
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    return iv + encrypted_data


def decrypt_with_camellia(encrypted_data_with_iv: bytes, key: bytes, crypto_config: dict) -> bytes:
    """Расшифровка данных алгоритмом Camellia в режиме CBC"""
    block_size = crypto_config['block_size']
    
    iv = encrypted_data_with_iv[:block_size]
    encrypted_data = encrypted_data_with_iv[block_size:]
    
    cipher = Cipher(algorithms.Camellia(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
    return unpad_data(decrypted_padded, block_size)