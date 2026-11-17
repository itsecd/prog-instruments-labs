"""Конфигурация приложения - загрузка и сохранение настроек в JSON файлы"""

import os
import json


def crypto_config(config_file: str) -> dict:
    """Загружает криптографические константы из JSON файла.
    
    Args:
        config_file: Путь к файлу с криптографическими параметрами.
    
    Returns:
        dict: Словарь с параметрами.
    
    Raises:
        FileNotFoundError: Если файл конфигурации не существует.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Файл {config_file} не найден.\n"
            f"Создайте его с содержимым:\n"
            f'{{\n'
            f'    "block_size": 16,\n'
            f'    "symmetric_key_size": 32,\n'
            f'    "rsa_key_size": 2048,\n'
            f'    "rsa_public_exponent": 65537\n'
            f'}}'
        )
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def settings(settings_file: str) -> dict:
    """Загружает пользовательские настройки из JSON файла.
    
    Args:
        settings_file: Путь к файлу с настройками путей.
    
    Returns:
        dict: Словарь с путями к файлам.
    
    Raises:
        FileNotFoundError: Если файл настроек не существует.
    """
    if not os.path.exists(settings_file):
        raise FileNotFoundError(
            f"Файл {settings_file} не найден.\n"
            f"Создайте его с содержимым:\n"
            f'{{\n'
            f'    "initial_file": "data/plaintext.txt",\n'
            f'    "encrypted_file": "data/encrypted.bin",\n'
            f'    "decrypted_file": "data/decrypted.txt",\n'
            f'    "symmetric_key": "keys/symmetric.key",\n'
            f'    "encrypted_symmetric_key": "keys/encrypted_symmetric.key",\n'
            f'    "public_key": "keys/public.pem",\n'
            f'    "private_key": "keys/private.pem"\n'
            f'}}'
        )
    
    with open(settings_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_settings(settings_dict: dict, settings_file: str) -> None:
    """Сохраняет настройки путей в файл.
    
    Args:
        settings_dict: Словарь с настройками для сохранения.
        settings_file: Путь к файлу для сохранения.
    """
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings_dict, f, indent=4, ensure_ascii=False)
    print(f"[OK] Настройки сохранены в {settings_file}")