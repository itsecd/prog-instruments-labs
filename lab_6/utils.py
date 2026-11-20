import json
from typing import Dict, Any
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding, rsa
from cryptography.hazmat.primitives import hashes, serialization

def load_settings(filepath: str = 'settings.json') -> Dict[str, Any]:
    """Загружает конфигурационные настройки из JSON файла.

    Args:
        filepath (str): Путь к файлу конфигурации. По умолчанию 'settings.json'.

    Returns:
        Dict[str, Any]: Словарь со структурированными путями и параметрами лабы.
    """
    try:
        with open(filepath, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"Файл {filepath} не найден!")
        return 0
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON в файле {filepath}: {e}")
        return 0
    except Exception as e:
        print(f"Неизвестная ошибка при открытии файла '{filepath}': '{e}'")
        return 0

def get_asym_padding() -> asym_padding.OAEP:
    """Возвращает настроенную схему паддинга RSA-OAEP.

    Использует алгоритм хэширования SHA-256 для MGF1 и главного хэша.

    Returns:
        asym_padding.OAEP: Объект асимметричного паддинга для RSA.
    """
    return asym_padding.OAEP(
        mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )

def read_bytes_safe(filepath: str) -> bytes | None:
    """Безопасное чтение байтов из файла с обработкой его отсутствия."""
    try:
        with open(filepath, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Файл {filepath} не найден!")
        return None

def write_bytes_safe(filepath: str, data: bytes, error_msg: str) -> bool:
    """Безопасная запись байтов в файл с обработкой ошибок ввода-вывода."""
    try:
        with open(filepath, 'wb') as f:
            f.write(data)
        return True
    except IOError as e:
        print(f"{error_msg}: {e}")
        return False

def load_private_key(filepath: str) -> rsa.RSAPrivateKey | None:
    """Загрузка закрытого ключа RSA из файла PEM."""
    private_bytes = read_bytes_safe(filepath)
    if private_bytes is None:
        return None
    try:
        return serialization.load_pem_private_key(private_bytes, password=None)
    except ValueError:
        print(f"Неверный формат закрытого ключа в файле {filepath}!")
        return None

def save_public_key(public_key: rsa.RSAPublicKey, filepath: str) -> bool:
    """Сериализация и сохранение открытого ключа RSA в файл PEM."""
    try:
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return write_bytes_safe(filepath, pem, "Ошибка при сохранении публичного ключа")
    except Exception as e:
        print(f"Ошибка сериализации публичного ключа: {e}")
        return False

def save_private_key(private_key: rsa.RSAPrivateKey, filepath: str) -> bool:
    """Сериализация и сохранение закрытого ключа RSA в файл PEM."""
    try:
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        return write_bytes_safe(filepath, pem, "Ошибка при сохранении приватного ключа")
    except Exception as e:
        print(f"Ошибка сериализации закрытого ключа: {e}")
        return False