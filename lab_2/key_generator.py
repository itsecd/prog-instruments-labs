"""Модуль генерации ключей гибридной системы"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.symmetric import generate_camellia_key
from src.asymmetric import generate_rsa_keypair, save_public_key, save_private_key, encrypt_with_rsa
from src.utils import write_binary_file


def run_key_generation(public_key_path: str, private_key_path: str, 
                       symmetric_key_path: str, encrypted_symmetric_key_path: str,
                       crypto_config: dict) -> None:
    """Режим генерации ключей (п.1.1 - 1.4)"""
    print("\n" + "="*60)
    print("РЕЖИМ 1: ГЕНЕРАЦИЯ КЛЮЧЕЙ ГИБРИДНОЙ СИСТЕМЫ")
    print("="*60)
    
    print("\n[1.1] Генерация симметричного ключа (Camellia)...")
    symmetric_key = generate_camellia_key(crypto_config)
    write_binary_file(symmetric_key_path, symmetric_key)
    print(f"[OK] Симметричный ключ сохранен: {symmetric_key_path}")
    
    print("\n[1.2] Генерация асимметричных ключей (RSA)...")
    private_key, public_key = generate_rsa_keypair(crypto_config)
    
    print("\n[1.3] Сохранение ключей...")
    save_public_key(public_key, public_key_path)
    save_private_key(private_key, private_key_path)
    
    print("\n[1.4] Зашифрование симметричного ключа открытым ключом...")
    encrypted_key = encrypt_with_rsa(symmetric_key, public_key_path)
    write_binary_file(encrypted_symmetric_key_path, encrypted_key)
    print(f"[OK] Зашифрованный симметричный ключ сохранен: {encrypted_symmetric_key_path}")
    
    print("\n" + "="*60)
    print("[ГОТОВО] Генерация ключей завершена!")
    print("="*60)