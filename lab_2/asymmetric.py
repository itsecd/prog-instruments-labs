"""Модуль асимметричной криптографии (RSA)"""

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

from .utils import read_binary_file, write_binary_file


def generate_rsa_keypair(crypto_config: dict):
    """Генерация пары RSA ключей.
    
    Args:
        crypto_config: Словарь с криптографическими параметрами.
    """
    private_key = rsa.generate_private_key(
        public_exponent=crypto_config['rsa_public_exponent'],
        key_size=crypto_config['rsa_key_size']
    )
    return private_key, private_key.public_key()


def save_public_key(public_key, filepath: str) -> None:
    """Сохранение открытого ключа в PEM файл."""
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    write_binary_file(filepath, pem)
    print(f"[OK] Открытый ключ сохранен: {filepath}")


def save_private_key(private_key, filepath: str) -> None:
    """Сохранение закрытого ключа в PEM файл."""
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    write_binary_file(filepath, pem)
    print(f"[OK] Закрытый ключ сохранен: {filepath}")


def load_public_key(filepath: str):
    """Загрузка открытого ключа из PEM файла."""
    data = read_binary_file(filepath)
    return serialization.load_pem_public_key(data)


def load_private_key(filepath: str):
    """Загрузка закрытого ключа из PEM файла."""
    data = read_binary_file(filepath)
    return serialization.load_pem_private_key(data, password=None)


def encrypt_with_rsa(key_data: bytes, public_key_path: str) -> bytes:
    """Шифрование данных открытым RSA ключом."""
    public_key = load_public_key(public_key_path)
    return public_key.encrypt(
        key_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )


def decrypt_with_rsa(encrypted_data: bytes, private_key_path: str) -> bytes:
    """Расшифровка данных закрытым RSA ключом."""
    private_key = load_private_key(private_key_path)
    return private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )