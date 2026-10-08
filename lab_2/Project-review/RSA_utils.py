import os
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from typing import Tuple


def generate_keys(key_size: int, public_exponent: int) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """
    Создаёт пару ключей RSA: приватный и публичный.
    
    Args:
        key_size: Размер ключа в битах.
        public_exponent: Открытая экспонента.
    
    Returns:
        Кортеж (приватный_ключ, публичный_ключ).
    """
    try:
        private_key = rsa.generate_private_key(
            public_exponent=public_exponent,
            key_size=key_size
        )
        public_key = private_key.public_key()
        return private_key, public_key
    except ValueError as e:
        raise ValueError(f"Ошибка параметров RSA: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Не удалось создать ключи RSA: {e}") from e


def encrypt_with_public_key(data: bytes, public_key: rsa.RSAPublicKey) -> bytes:
    """
    Шифрует данные публичным ключом RSA с использованием OAEP.
    
    Args:
        data: Данные для шифрования.
        public_key: Публичный ключ RSA.
    
    Returns:
        Зашифрованные данные.
    """
    try:
        encrypted_data = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_data
    except ValueError as e:
        raise ValueError(f"Ошибка шифрования RSA: {e}") from e
    except TypeError as e:
        raise TypeError(f"Неверный тип ключа: {e}") from e
    except (InvalidSignature, UnsupportedAlgorithm) as e:
        raise RuntimeError(f"Криптографическая ошибка: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Неожиданная ошибка: {e}") from e


def decrypt_with_private_key(encrypted_data: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
    """
    Расшифровывает данные приватным ключом RSA.
    
    Args:
        encrypted_data: Зашифрованные данные.
        private_key: Приватный ключ RSA.
    
    Returns:
        Расшифрованные данные.
    """
    try:
        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data
    except ValueError as e:
        raise ValueError(f"Ошибка расшифрования RSA: {e}") from e
    except TypeError as e:
        raise TypeError(f"Неверный тип ключа: {e}") from e
    except (InvalidSignature, UnsupportedAlgorithm) as e:
        raise RuntimeError(f"Криптографическая ошибка: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Неожиданная ошибка: {e}") from e


def serialize_public_key(public_key: rsa.RSAPublicKey, file_path: str) -> None:
    """
    Сохраняет публичный ключ в PEM файл.
    
    Args:
        public_key: Публичный ключ RSA.
        file_path: Путь для сохранения.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pem_key = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        _write_binary_file(file_path, pem_key)
    except OSError as e:
        raise OSError(f"Не удалось сохранить публичный ключ: {e}") from e
    except TypeError as e:
        raise TypeError(f"Неверный тип ключа: {e}") from e


def serialize_private_key(private_key: rsa.RSAPrivateKey, file_path: str) -> None:
    """
    Сохраняет приватный ключ в PEM файл без шифрования.
    
    Args:
        private_key: Приватный ключ RSA.
        file_path: Путь для сохранения.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pem_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        _write_binary_file(file_path, pem_key)
    except OSError as e:
        raise OSError(f"Не удалось сохранить приватный ключ: {e}") from e
    except TypeError as e:
        raise TypeError(f"Неверный тип ключа: {e}") from e


def load_public_key(file_path: str) -> rsa.RSAPublicKey:
    """
    Загружает публичный ключ из PEM файла.
    
    Args:
        file_path: Путь к PEM файлу.
    
    Returns:
        Загруженный публичный ключ.
    """
    try:
        pem_data = _read_binary_file(file_path)
        public_key = load_pem_public_key(pem_data)
        return public_key
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл ключа не найден: {file_path}") from e
    except ValueError as e:
        raise ValueError(f"Ошибка загрузки PEM ключа: {e}") from e


def load_private_key(file_path: str) -> rsa.RSAPrivateKey:
    """
    Загружает приватный ключ из PEM файла.
    
    Args:
        file_path: Путь к PEM файлу.
    
    Returns:
        Загруженный приватный ключ.
    """
    try:
        pem_data = _read_binary_file(file_path)
        private_key = load_pem_private_key(pem_data, password=None)
        return private_key
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл ключа не найден: {file_path}") from e
    except ValueError as e:
        raise ValueError(f"Ошибка загрузки приватного ключа: {e}") from e


def _write_binary_file(file_path: str, data: bytes) -> None:
    """Внутренняя функция для записи бинарных данных."""
    try:
        with open(file_path, "wb") as f:
            f.write(data)
    except OSError as e:
        raise OSError(f"Ошибка записи в файл {file_path}: {e}") from e


def _read_binary_file(file_path: str) -> bytes:
    """Внутренняя функция для чтения бинарных данных."""
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл не найден: {file_path}") from e
    except OSError as e:
        raise OSError(f"Ошибка чтения файла {file_path}: {e}") from e