from typing import Dict, Any
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import utils

def decrypt_data(settings: Dict[str, Any]) -> None:
    """Выполняет дешифрование ранее зашифрованного файла.

    Восстанавливает ключ Camellia через закрытый ключ RSA, считывает IV 
    из первых 16 байт зашифрованного файла, расшифровывает основной шифротекст 
    в режиме CBC и производит депаддинг ANSIX923.

    Args:
        settings (Dict[str, Any]): Конфигурационный словарь с путями к файлам.
    """
    print("Запуск режима дешифрования...")
    
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

    file_content = utils.read_bytes_safe(settings['encrypted_file'])
    if file_content is None:
        return

    if len(file_content) < 16:
        print("Ошибка: файл слишком короткий для извлечения IV.")
        return

    iv = file_content[:16]
    c_text = file_content[16:]
    
    cipher = Cipher(algorithms.Camellia(sym_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_dc_text = decryptor.update(c_text) + decryptor.finalize()

    unpadder = sym_padding.ANSIX923(128).unpadder()
    try:
        dc_text = unpadder.update(padded_dc_text) + unpadder.finalize()
    except Exception as e:
        print(f"Ошибка удаления дополнения (Padding error): {e}")
        return

    if utils.write_bytes_safe(settings['decrypted_file'], dc_text, "Ошибка при сохранении дешифрованного файла"):
        print(f"Данные успешно дешифрованы и сохранены в: {settings['decrypted_file']}.")
        print("Дешифрование завершено!\n")