import os
from typing import Dict, Any
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import utils

def encrypt_data(settings: Dict[str, Any]) -> None:
    """Выполняет гибридное шифрование текстового файла.

    Расшифровывает сессионный ключ Camellia с помощью закрытого ключа RSA,
    дополняет исходный текст паддингом ANSIX923 и шифрует его в режиме CBC.
    Вектор инициализации (IV) сохраняется в первые 16 байт итогового файла.

    Args:
        settings (Dict[str, Any]): Конфигурационный словарь с путями к файлам.
    """
    print("Запуск режима шифрования...")
    
    private_key = utils.load_private_key(settings['secret_key'])
    if private_key is None:
        return
    
    enc_sym_key = utils.read_bytes_safe(settings['symmetric_key'])
    if enc_sym_key is None:
        return
    
    try:
        sym_key = private_key.decrypt(enc_sym_key, utils.get_asym_padding())
        print("Симметричный ключ успешно расшифрован.")
    except Exception as e:
        print(f"Ошибка расшифрования симметричного ключа: {e}")
        return
    
    text = utils.read_bytes_safe(settings['initial_file'])
    if text is None:
        return

    padder = sym_padding.ANSIX923(128).padder() 
    padded_text = padder.update(text) + padder.finalize()

    iv = os.urandom(16) 
    cipher = Cipher(algorithms.Camellia(sym_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    c_text = encryptor.update(padded_text) + encryptor.finalize()

    if utils.write_bytes_safe(settings['encrypted_file'], iv + c_text, "Ошибка при сохранении зашифрованного файла"):
        print(f"Данные зашифрованы алгоритмом Camellia (с IV) и сохранены в: {settings['encrypted_file']}.")
        print("Шифрование завершено!\n")