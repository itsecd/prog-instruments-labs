from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.asymmetric import rsa
from crypto.asymmetric import AsymmetricCrypto
from utils.file_utils import FileManager


class KeyManager:
    """Управление RSA ключами."""
    
    def __init__(self, config: Dict[str, Any], file_manager: FileManager, private_path: str, public_path: str):
        """
        Инициализирует менеджер ключей с путями к RSA ключам.
        
        Args:
            config (Dict[str, Any]): конфигурация
            file_manager (FileManager): для файловых операций
            private_path (str): путь к приватному ключу
            public_path (str): путь к публичному ключу
        """
        self.config = config
        self.file_manager = file_manager
        self.asymmetric_crypto = AsymmetricCrypto(config)
        self._private_path = private_path
        self._public_path = public_path
    
    def ensure_rsa_keys_exist(self) -> None:
        """Проверяет существование ключей, генерирует если нет."""
        private_exists = self.file_manager.file_exists(self._private_path)
        public_exists = self.file_manager.file_exists(self._public_path)
        
        match (private_exists, public_exists):
            case (True, True):
                return
            case _:
                self._generate_and_save_rsa_keys()
    
    def _generate_and_save_rsa_keys(self) -> None:
        """Генерирует и сохраняет RSA ключи."""
        private_key, public_key = self.asymmetric_crypto.generate_rsa_keypair()
        private_bytes = self.asymmetric_crypto.save_private_key_to_bytes(private_key)
        public_bytes = self.asymmetric_crypto.save_public_key_to_bytes(public_key)
        self.file_manager.write_file(self._private_path, private_bytes, binary=True)
        self.file_manager.write_file(self._public_path, public_bytes, binary=True)
    
    def load_public_key(self) -> rsa.RSAPublicKey:
        """
        Загружает публичный RSA ключ из файла.
        
        Returns:
            rsa.RSAPublicKey: публичный ключ
        """
        public_bytes = self.file_manager.read_file(self._public_path, binary=True)
        return self.asymmetric_crypto.load_public_key(public_bytes)
    
    def load_private_key(self) -> rsa.RSAPrivateKey:
        """
        Загружает приватный RSA ключ из файла.
        
        Returns:
            rsa.RSAPrivateKey: приватный ключ
        """
        private_bytes = self.file_manager.read_file(self._private_path, binary=True)
        return self.asymmetric_crypto.load_private_key(private_bytes)