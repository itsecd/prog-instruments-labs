import argparse
from typing import Dict, Any
from utils import load_settings
from keygen import generate_keys
from encrypt import encrypt_data
from decrypt import decrypt_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Гибридная криптосистема (RSA + Camellia)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen', '--generation', action='store_true', help='Запускает режим генерации ключей')
    group.add_argument('-enc', '--encryption', action='store_true', help='Запускает режим шифрования')
    group.add_argument('-dec', '--decryption', action='store_true', help='Запускает режим дешифрования')
    parser.add_argument('-s', '--size', type=int, choices=[128, 192, 256], default=256, help='Длина ключа Camellia в битах (128, 192 или 256)')

    parser.add_argument('--public_key', type=str, help='Путь к публичному ключу RSA')
    parser.add_argument('--secret_key', type=str, help='Путь к секретному ключу RSA')   
    parser.add_argument('--symmetric_key', type=str, help='Путь к зашифрованному ключу Camellia')
    parser.add_argument('--initial_file', type=str, help='Путь к исходному файлу')
    parser.add_argument('--encrypted_file', type=str, help='Путь к зашифрованному файлу')
    parser.add_argument('--decrypted_file', type=str, help='Путь к расшифрованному файлу')

    args = parser.parse_args()
    config: Dict[str, Any] = load_settings()

    if args.public_key: config['public_key'] = args.public_key
    if args.secret_key: config['secret_key'] = args.secret_key
    if args.symmetric_key: config['symmetric_key'] = args.symmetric_key
    if args.initial_file: config['initial_file'] = args.initial_file
    if args.encrypted_file: config['encrypted_file'] = args.encrypted_file
    if args.decrypted_file: config['decrypted_file'] = args.decrypted_file

    mode = 'gen' if args.generation else 'enc' if args.encryption else 'dec'
    
    match mode:
        case 'gen':
            generate_keys(config, args.size)
        case 'enc':
            encrypt_data(config)
        case 'dec':
            decrypt_data(config)