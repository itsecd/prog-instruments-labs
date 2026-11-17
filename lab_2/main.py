#!/usr/bin/env python3
"""Гибридная криптосистема с использованием RSA и Camellia."""

import argparse
import os
import sys

from src.config import crypto_config, settings, save_settings
from src.key_generator import run_key_generation
from src.encryptor import run_encryption
from src.decryptor import run_decryption


def create_directories() -> None:
    """Создает стандартные директории для работы программы."""
    for directory in ['data', 'keys']:
        os.makedirs(directory, exist_ok=True)


def configure_paths(settings_dict: dict, settings_file: str) -> dict:
    """Позволяет пользователю интерактивно настроить пути к файлам."""
    print("\n--- НАСТРОЙКА ПУТЕЙ ---")
    print("Текущие значения (Enter - оставить без изменений):")
    
    for key in settings_dict.keys():
        new_value = input(f"  {key} [{settings_dict[key]}]: ").strip()
        if new_value:
            settings_dict[key] = new_value
    
    save_settings(settings_dict, settings_file)
    return settings_dict


def print_menu() -> None:
    """Выводит главное меню программы."""
    print("\n" + "="*60)
    print("ГИБРИДНАЯ КРИПТОСИСТЕМА (RSA + Camellia)")
    print("Лабораторная работа №3")
    print("="*60)
    print("\nРежимы работы:")
    print("  1 - Генерация ключей ")
    print("  2 - Шифрование данных ")
    print("  3 - Дешифрование данных ")
    print("  4 - Настройка путей")
    print("  0 - Выход")
    print("-"*60)


def interactive_mode(crypto_config_file: str, settings_file: str) -> None:
    """Запускает интерактивный режим с текстовым меню."""
    create_directories()
    
    # Загрузка криптографических параметров
    try:
        crypto_params = crypto_config(crypto_config_file)
    except FileNotFoundError as e:
        print(f"\n[ОШИБКА] {e}")
        return
    
    # Загрузка настроек путей
    try:
        settings_dict = settings(settings_file)
    except FileNotFoundError as e:
        print(f"\n[ОШИБКА] {e}")
        return
    
    while True:
        print_menu()
        choice = input("\nВаш выбор: ").strip()
        
        match choice:
            case '1':
                print("\n--- ГЕНЕРАЦИЯ КЛЮЧЕЙ ---")
                run_key_generation(
                    public_key_path=settings_dict['public_key'],
                    private_key_path=settings_dict['private_key'],
                    symmetric_key_path=settings_dict['symmetric_key'],
                    encrypted_symmetric_key_path=settings_dict['encrypted_symmetric_key'],
                    crypto_config=crypto_params
                )
                input("\nНажмите Enter для продолжения...")
            
            case '2':
                print("\n--- ШИФРОВАНИЕ ДАННЫХ ---")
                
                if not os.path.exists(settings_dict['encrypted_symmetric_key']):
                    print(f"\n[ОШИБКА] Зашифрованный симметричный ключ не найден: {settings_dict['encrypted_symmetric_key']}")
                    print("Сначала выполните режим 1 (Генерация ключей).")
                    input("\nНажмите Enter для продолжения...")
                    continue
                
                run_encryption(
                    input_file=settings_dict['initial_file'],
                    private_key_path=settings_dict['private_key'],
                    encrypted_symmetric_key_path=settings_dict['encrypted_symmetric_key'],
                    output_file=settings_dict['encrypted_file'],
                    crypto_config=crypto_params
                )
                input("\nНажмите Enter для продолжения...")
            
            case '3':
                print("\n--- ДЕШИФРОВАНИЕ ДАННЫХ ---")
                
                if not os.path.exists(settings_dict['encrypted_symmetric_key']):
                    print(f"\n[ОШИБКА] Зашифрованный симметричный ключ не найден: {settings_dict['encrypted_symmetric_key']}")
                    print("Сначала выполните режим 1 (Генерация ключей).")
                    input("\nНажмите Enter для продолжения...")
                    continue
                
                if not os.path.exists(settings_dict['encrypted_file']):
                    print(f"\n[ОШИБКА] Зашифрованный файл не найден: {settings_dict['encrypted_file']}")
                    print("Сначала выполните режим 2 (Шифрование).")
                    input("\nНажмите Enter для продолжения...")
                    continue
                
                run_decryption(
                    encrypted_file=settings_dict['encrypted_file'],
                    private_key_path=settings_dict['private_key'],
                    encrypted_symmetric_key_path=settings_dict['encrypted_symmetric_key'],
                    output_file=settings_dict['decrypted_file'],
                    crypto_config=crypto_params
                )
                input("\nНажмите Enter для продолжения...")
            
            case '4':
                settings_dict = configure_paths(settings_dict, settings_file)
            
            case '0':
                print("\nДо свидания!")
                break
            
            case _:
                print("\n[ОШИБКА] Неверный выбор. Пожалуйста, выберите 0-4.")
                input("Нажмите Enter для продолжения...")


def main() -> None:
    """Основная точка входа в программу."""
    parser = argparse.ArgumentParser(
        description='Гибридная криптосистема (RSA + Camellia)',
        epilog='''
Примеры использования:
  python main.py                                                   # Интерактивный режим
  python main.py --crypto-config my_crypto.json --settings my_settings.json
  python main.py --interactive                                     # Интерактивный режим
  python main.py --help
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--interactive', action='store_true',
                        help='Запуск в интерактивном режиме')
    parser.add_argument('--crypto-config', type=str, default='crypto_config.json',
                        help='Путь к файлу с криптографическими параметрами (по умолчанию: crypto_config.json)')
    parser.add_argument('--settings', type=str, default='settings.json',
                        help='Путь к файлу с настройками путей (по умолчанию: settings.json)')
    
    args = parser.parse_args()
    
    if args.interactive or len(sys.argv) == 1:
        interactive_mode(args.crypto_config, args.settings)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()