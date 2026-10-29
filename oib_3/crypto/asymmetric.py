from typing import Dict, Any, Tuple
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend


class AsymmetricCrypto:
    """RSA асимметричное шифрование с OAEP дополнением."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): словарь конфигурации
        """
        crypto = config.get('crypto')
        if crypto is None:
            raise KeyError("Секция 'crypto' не найдена")

        rsa_section = crypto.get('rsa')
        if rsa_section is None:
            raise KeyError("Секция 'rsa' не найдена")

        self._rsa_key_size = rsa_section[0]
        self._public_exponent = rsa_section[1]
        padding_section = rsa_section[2]
        self._label = padding_section[3]
        
        if isinstance(self._label, str):
            self._label = self._label.encode('utf-8')
        
        self.backend = default_backend()
    
    def generate_rsa_keypair(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Создает пару ключей для шифрования RSA
        Returns:
            Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]: (приватный, публичный)
        """
        private_key = rsa.generate_private_key(
            public_exponent=self._public_exponent,
            key_size=self._rsa_key_size,
            backend=self.backend
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def _get_oaep_padding(self) -> padding.OAEP:
        """Создает и возвращает схему дополнения OAEP."""
        return padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=self._label
        )
    
    def encrypt_with_public_key(self, data: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        """
        Шифрует данные открытым ключом
        Args:
            data (bytes): данные для шифрования
            public_key (rsa.RSAPublicKey): публичный ключ
        
        Returns:
            bytes: зашифрованные данные
        """
        if not data:
            raise ValueError("Нет данных для шифрования")
        
        return public_key.encrypt(data, self._get_oaep_padding())
    
    def decrypt_with_private_key(self, encrypted_data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """
        Дешифрует данные закрытым ключом
        Args:
            encrypted_data (bytes): зашифрованные данные
            private_key (rsa.RSAPrivateKey): приватный ключ
        
        Returns:
            bytes: расшифрованные данные
        """
        
        if not encrypted_data:
            raise ValueError("Нет данных для расшифрования")
        
        return private_key.decrypt(encrypted_data, self._get_oaep_padding())
    
    def load_private_key(self, key_data: bytes) -> rsa.RSAPrivateKey:
        """
        Загружает приватный ключ из PEM байтов
        Args:
            key_data (bytes): PEM байты
        
        Returns:
            rsa.RSAPrivateKey: приватный ключ
        """
        return serialization.load_pem_private_key(key_data, password=None, backend=self.backend)
    
    def load_public_key(self, key_data: bytes) -> rsa.RSAPublicKey:
        """
        Загружает публичный ключ из PEM байтов.
        Args:
            key_data (bytes): PEM байты
        
        Returns:
            rsa.RSAPublicKey: публичный ключ
        """
        return serialization.load_pem_public_key(key_data, backend=self.backend)
    
    def save_private_key_to_bytes(self, private_key: rsa.RSAPrivateKey) -> bytes:
        """
        Сохраняет приватный ключ в PEM формат
        Args:
            private_key (rsa.RSAPrivateKey): приватный ключ
        
        Returns:
            bytes: ключ в PEM формате
        """
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def save_public_key_to_bytes(self, public_key: rsa.RSAPublicKey) -> bytes:
        """
        Сохраняет публичный ключ в PEM формат
        Args:
            public_key (rsa.RSAPublicKey): публичный ключ
        
        Returns:
            bytes: ключ в PEM формате
        """
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )