import os
from typing import Dict, Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


class SymmetricCipher:
    """Базовый класс для симметричных шифров."""
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Шифрует открытый текст симметричным ключом."""
        raise NotImplementedError
    
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Расшифровывает шифротекст симметричным ключом."""
        raise NotImplementedError
    
    def generate_key(self) -> bytes:
        """Генерирует случайный симметричный ключ."""
        raise NotImplementedError
    
    def get_key_size(self) -> int:
        """Возвращает размер ключа в байтах."""
        raise NotImplementedError


class SEEDCipher(SymmetricCipher):
    """SEED шифрование, CBC режим, PKCS7 padding."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует SEED шифр с параметрами из конфигурации.
        
        Args:
            config (Dict[str, Any]): конфигурация
        """
        crypto_section = config.get('crypto')
        match crypto_section:
            case None:
                raise ValueError("Секция 'crypto' не найдена")
        
        seed_section = crypto_section.get('seed')
        match seed_section:
            case None:
                raise ValueError("Секция 'seed' не найдена")
            case [16, 16]:
                self._key_size = 16
                self._block_size = 16
            case _:
                raise ValueError("SEED требует [16, 16]")
        
        self.backend = default_backend()
    
    def generate_key(self) -> bytes:
        """Генерирует случайный 16-байтный ключ для SEED."""
        return os.urandom(self._key_size)
    
    def get_key_size(self) -> int:
        """Возвращает размер ключа SEED (16 байт)."""
        return self._key_size
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """
        Шифрует данные алгоритмом SEED в режиме CBC с PKCS7 padding.
        
        Args:
            plaintext (bytes): открытый текст
            key (bytes): ключ 16 байт
        
        Returns:
            bytes: IV + зашифрованные данные
        """
        match plaintext:
            case b'':
                raise ValueError("Нет данных для шифрования")
        
        iv = os.urandom(self._block_size)
        padder = padding.PKCS7(self._block_size * 8).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.SEED(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        return iv + encrypted
    
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """
        Расшифровывает данные алгоритмом SEED в режиме CBC.
        
        Args:
            ciphertext (bytes): IV + зашифрованные данные
            key (bytes): ключ 16 байт
        
        Returns:
            bytes: расшифрованный текст
        """
        match ciphertext:
            case bytes() if len(ciphertext) < 16:
                raise ValueError("Шифротекст слишком короткий")
        
        iv = ciphertext[:16]
        encrypted = ciphertext[16:]
        
        match encrypted:
            case b'':
                raise ValueError("Нет зашифрованных данных")
        
        cipher = Cipher(algorithms.SEED(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(decrypted_padded) + unpadder.finalize()
        return plaintext


class ChaCha20Cipher(SymmetricCipher):
    """ChaCha20-Poly1305 аутентифицированное шифрование."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует ChaCha20 шифр с параметрами из конфигурации.
        
        Args:
            config (Dict[str, Any]): конфигурация
        """
        crypto_section = config.get('crypto')
        match crypto_section:
            case None:
                raise ValueError("Секция 'crypto' не найдена")
        
        chacha_section = crypto_section.get('chacha20')
        match chacha_section:
            case None:
                raise ValueError("Секция 'chacha20' не найдена")
            case [32, 12]:
                self._key_size = 32
                self._nonce_size = 12
            case _:
                raise ValueError("ChaCha20 требует [32, 12]")
    
    def generate_key(self) -> bytes:
        """Генерирует случайный 32-байтный ключ для ChaCha20."""
        return os.urandom(self._key_size)
    
    def get_key_size(self) -> int:
        """Возвращает размер ключа ChaCha20 (32 байта)."""
        return self._key_size
    
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """
        Шифрует данные алгоритмом ChaCha20-Poly1305 с аутентификацией.
        
        Args:
            plaintext (bytes): открытый текст
            key (bytes): ключ 32 байта
        
        Returns:
            bytes: nonce + зашифрованные данные
        """
        match plaintext:
            case b'':
                raise ValueError("Нет данных для шифрования")
        
        nonce = os.urandom(self._nonce_size)
        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(nonce, plaintext, None)
        return nonce + ciphertext
    
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """
        Расшифровывает данные алгоритмом ChaCha20-Poly1305 с проверкой аутентификации.
        
        Args:
            ciphertext (bytes): nonce + зашифрованные данные
            key (bytes): ключ 32 байта
        
        Returns:
            bytes: расшифрованный текст
        """
        match ciphertext:
            case bytes() if len(ciphertext) < 12:
                raise ValueError("Шифротекст слишком короткий")
        
        nonce = ciphertext[:12]
        encrypted = ciphertext[12:]
        
        match encrypted:
            case b'':
                raise ValueError("Нет зашифрованных данных")
        
        chacha = ChaCha20Poly1305(key)
        plaintext = chacha.decrypt(nonce, encrypted, None)
        
        match plaintext:
            case b'':
                raise RuntimeError("Результат расшифрования пуст")
        
        return plaintext