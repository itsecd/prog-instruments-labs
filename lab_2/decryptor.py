"""Модуль дешифрования данных гибридной системой"""

import os
import sys
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import read_binary_file, write_text_file
from src.asymmetric import decrypt_with_rsa
from src.symmetric import decrypt_with_camellia


def run_decryption(encrypted_file: str, private_key_path: str, 
                   encrypted_symmetric_key_path: str, output_file: str,
                   crypto_config: dict) -> None:
    """Режим дешифрования (п.3.1 - 3.2) с обработкой ошибок"""
    print("\n" + "="*60)
    print("РЕЖИМ 3: ДЕШИФРОВАНИЕ ДАННЫХ ГИБРИДНОЙ СИСТЕМОЙ")
    print("="*60)
    
    error_occurred = False
    
    print("\n[3.1] Расшифровка симметричного ключа...")
    try:
        encrypted_key = read_binary_file(encrypted_symmetric_key_path)
        symmetric_key = decrypt_with_rsa(encrypted_key, private_key_path)
        print(f"[OK] Симметричный ключ получен ({len(symmetric_key)} байт)")
    except ValueError as e:
        print(f"[ОШИБКА] Не удалось расшифровать симметричный ключ: {e}")
        print("Возможные причины: неверный закрытый ключ или поврежден зашифрованный ключ.")
        error_occurred = True
        return
    except Exception as e:
        print(f"[ОШИБКА] Непредвиденная ошибка при расшифровке ключа: {e}")
        error_occurred = True
        return
    
    print(f"\n[3.2] Чтение зашифрованного файла: {encrypted_file}")
    try:
        encrypted_data = read_binary_file(encrypted_file)
        print(f"[OK] Прочитано {len(encrypted_data)} байт")
    except FileNotFoundError:
        print(f"[ОШИБКА] Файл не найден: {encrypted_file}")
        error_occurred = True
        return
    except Exception as e:
        print(f"[ОШИБКА] Ошибка чтения файла: {e}")
        error_occurred = True
        return
    
    print("\n[3.3] Расшифровка данных Camellia (CBC)...")
    try:
        decrypted_data = decrypt_with_camellia(encrypted_data, symmetric_key, crypto_config)
        print("[OK] Расшифровка выполнена успешно")
    except ValueError as e:
        print(f"[ОШИБКА] Ошибка при расшифровке: {e}")
        print("Возможные причины:")
        print("  - Зашифрованный файл поврежден (изменены байты)")
        print("  - Неверный симметричный ключ")
        print("  - Неправильный padding (данные были изменены)")
        error_occurred = True
        return
    except Exception as e:
        print(f"[ОШИБКА] Непредвиденная ошибка при расшифровке: {e}")
        error_occurred = True
        return
    
    print(f"\n[3.4] Сохранение расшифрованного файла: {output_file}")
    try:
        write_text_file(output_file, decrypted_data)
        print("[OK] Файл сохранен")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка сохранения файла: {e}")
        error_occurred = True
        return
    
    print("\n" + "="*60)
    if error_occurred:
        print("[НЕУДАЧА] Дешифрование завершено с ошибками!")
    else:
        print("[ГОТОВО] Дешифрование завершено успешно!")
    print("="*60)
    
    if not error_occurred:
        try:
            preview = decrypted_data[:200].decode('utf-8')
            print(f"\n[ПРЕВЬЮ РАСШИФРОВАННОГО ТЕКСТА]:")
            print("-" * 50)
            print(preview + ("..." if len(decrypted_data) > 200 else ""))
            print("-" * 50)
        except UnicodeDecodeError:
            print("\n[ПРЕДУПРЕЖДЕНИЕ] Расшифрованные данные не являются текстом UTF-8")