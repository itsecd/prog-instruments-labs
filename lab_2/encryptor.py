"""Модуль шифрования данных гибридной системой"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import read_text_file, write_binary_file
from src.asymmetric import decrypt_with_rsa
from src.symmetric import encrypt_with_camellia


def run_encryption(input_file: str, private_key_path: str, 
                   encrypted_symmetric_key_path: str, output_file: str,
                   crypto_config: dict) -> None:
    """Режим шифрования (п.2.1 - 2.2)"""
    print("\n" + "="*60)
    print("РЕЖИМ 2: ШИФРОВАНИЕ ДАННЫХ ГИБРИДНОЙ СИСТЕМОЙ")
    print("="*60)
    
    print("\n[2.1] Расшифровка симметричного ключа...")
    with open(encrypted_symmetric_key_path, 'rb') as f:
        encrypted_key = f.read()
    symmetric_key = decrypt_with_rsa(encrypted_key, private_key_path)
    print(f"[OK] Симметричный ключ получен ({len(symmetric_key)} байт)")
    
    print(f"\n[2.2] Чтение исходного файла: {input_file}")
    plaintext = read_text_file(input_file)
    print(f"[OK] Прочитано {len(plaintext)} байт")
    
    print("[2.2] Шифрование данных Camellia (CBC)...")
    encrypted_data = encrypt_with_camellia(plaintext, symmetric_key, crypto_config)
    
    print(f"[2.2] Сохранение зашифрованного файла: {output_file}")
    write_binary_file(output_file, encrypted_data)
    
    print("\n" + "="*60)
    print("[ГОТОВО] Шифрование завершено!")
    print("="*60)