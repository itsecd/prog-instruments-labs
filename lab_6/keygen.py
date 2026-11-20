import os
from typing import Dict, Any
from cryptography.hazmat.primitives.asymmetric import rsa
import utils

def generate_keys(settings: Dict[str, Any], key_size_bits: int = 256) -> None:
    """Генерирует ключи для гибридной системы и сериализует их на диск.

    Создает случайный симметричный ключ Camellia заданной длины и пару 
    ключей RSA (2048 бит). Зашифровывает ключ Camellia открытым ключом RSA.

    Args:
        settings (Dict[str, Any]): Конфигурационный словарь с путями к файлам.
        key_size_bits (int): Длина ключа Camellia в битах (128, 192 или 256).
    """
    print("Запуск генерации ключей...")
    
    if key_size_bits not in [128, 192, 256]:
        print("Ошибка: длина ключа Camellia должна быть 128, 192 или 256 бит.")
        return
    sym_key = os.urandom(key_size_bits // 8)
    print(f"Сгенерирован симметричный ключ Camellia ({key_size_bits} бит).")

    keys = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    private_key = keys
    public_key = keys.public_key()
    print("Сгенерирована пара ключей RSA (2048 бит).")

    if not utils.save_public_key(public_key, settings['public_key']) or \
       not utils.save_private_key(private_key, settings['secret_key']):
        return
    print(f"Асимметричные ключи сохранены в: {settings['public_key']} и {settings['secret_key']}.")

    enc_sym_key = public_key.encrypt(sym_key, utils.get_asym_padding())
    
    if utils.write_bytes_safe(settings['symmetric_key'], enc_sym_key, "Ошибка при сохранении ключей"):
        print(f"Симметричный ключ зашифрован RSA и сохранен в: {settings['symmetric_key']}.")
        print("Генерация ключей завершена!\n")