from typing import Dict, Any, Tuple

from crypto.symmetric import SEEDCipher, ChaCha20Cipher
from crypto.asymmetric import AsymmetricCrypto


class HybridCrypto:
    """Гибридное шифрование: RSA + SEED/ChaCha20."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): конфигурация
        """
        self.config = config
        self.asymmetric = AsymmetricCrypto(config)
        self._cipher_map = {
            'SEED': SEEDCipher(config),
            'ChaCha20': ChaCha20Cipher(config)
        }
    
    def get_cipher(self, algorithm_name: str):
        """
        Возвращает экземпляр симметричного шифра
        Args:
            algorithm_name (str): 'SEED' или 'ChaCha20'
        
        Returns:
            SymmetricCipher: экземпляр шифра
        """
        match algorithm_name:
            case 'SEED':
                return self._cipher_map['SEED']
            case 'ChaCha20':
                return self._cipher_map['ChaCha20']
            case _:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def encrypt_hybrid(self, plaintext: bytes, algorithm_name: str, public_key: Any, custom_key: bytes = None) -> Tuple[bytes, bytes]:
        """
        Выполняет гибридное шифрование: данные шифруются симметрично, ключ - RSA
        Args:
            plaintext (bytes): открытый текст
            algorithm_name (str): 'SEED' или 'ChaCha20'
            public_key (Any): RSA публичный ключ
            custom_key (bytes, optional): пользовательский ключ
        
        Returns:
            Tuple[bytes, bytes]: (шифротекст, зашифрованный_ключ)
        """
        cipher = self.get_cipher(algorithm_name)
        
        match custom_key:
            case None:
                symmetric_key = cipher.generate_key()
            case _:
                if len(custom_key) != cipher.get_key_size():
                    raise ValueError(f"Неверный размер ключа для {algorithm_name}")
                symmetric_key = custom_key
        
        encrypted_data = cipher.encrypt(plaintext, symmetric_key)
        encrypted_key = self.asymmetric.encrypt_with_public_key(symmetric_key, public_key)
        return encrypted_data, encrypted_key
    
    def decrypt_hybrid(self, encrypted_data: bytes, encrypted_key: bytes, algorithm_name: str, private_key: Any) -> bytes:
        """
        Выполняет гибридное расшифрование: RSA расшифровывает ключ, затем данные симметричный
        Args:
            encrypted_data (bytes): зашифрованные данные
            encrypted_key (bytes): зашифрованный ключ
            algorithm_name (str): 'SEED' или 'ChaCha20'
            private_key (Any): RSA приватный ключ
        
        Returns:
            bytes: расшифрованный текст
        """
        cipher = self.get_cipher(algorithm_name)
        symmetric_key = self.asymmetric.decrypt_with_private_key(encrypted_key, private_key)
        plaintext = cipher.decrypt(encrypted_data, symmetric_key)
        return plaintext