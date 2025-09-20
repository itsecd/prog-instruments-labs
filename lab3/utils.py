import json
import os

from cryptography.hazmat.primitives import serialization

def load_settings(settings_path='settings.json'):
    """ Загружает настройки из JSON файла с обработкой ошибок. """
    try:
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Файл настроек {settings_path} не найден")
        with open(settings_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Ошибка при парсинге JSON: {e}")
        raise
    except PermissionError as e:
        print(f"Нет доступа к файлу {settings_path}: {e}")
        raise
    except Exception as e:
        print(f"Неожиданная ошибка при загрузке настроек: {e}")
        raise

def save_public_key(key, path):
    """Записывает публичный ключ в файл."""
    try:
        with open(path, 'wb') as f:
            f.write(
                key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            )
    except Exception as e:
        print(f"Ошибка при сохранении публичного ключа: {e}")
        raise

def save_private_key(key, path):
    """Записывает приватный ключ в файл."""
    try:
        with open(path, 'wb') as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                )
            )
    except Exception as e:
        print(f"Ошибка при сохранении приватного ключа: {e}")
        raise

def serialize_key(key, path, key_type):
    """ Сериализует ключ в файл, если файл не существует. """
    if os.path.exists(path):
        print(f"Файл {path} уже существует, пропуск сериализации.")
        return

    print(f"Сериализация ключа {key_type} в {path}...")
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    match key_type:
        case 'symmetric':
            with open(path, 'wb') as f:
                f.write(key)
        case 'public':
            save_public_key(key, path)
        case 'private':
            save_private_key(key, path)
        case _:
            raise ValueError(f"Неверный тип ключа: {key_type}")
