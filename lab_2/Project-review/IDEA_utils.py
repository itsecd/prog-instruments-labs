import os
from cryptography.hazmat.primitives.ciphers import algorithms, modes, Cipher
from cryptography.hazmat.primitives import padding


def generate_key_for_IDEA(key_size: int) -> bytes:
    """
    Создаёт ключ для алгоритма IDEA.
    
    Args:
        key_size: Размер ключа в байтах, должен быть 16.
    
    Returns:
        Сгенерированный ключ из 16 байт.
    """
    match key_size:
        case 16:
            try:
                key = os.urandom(key_size)
                if all(b == 0 for b in key):
                    raise RuntimeError("Сгенерирован нулевой ключ")
                return key
            except NotImplementedError as e:
                raise RuntimeError(f"Система не поддерживает RNG: {e}") from e
            except OSError as e:
                raise RuntimeError(f"Ошибка при чтении из /dev/urandom: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Неожиданная ошибка: {e}") from e
        case _:
            raise ValueError(f"IDEA требует ключ 16 байт. Получено: {key_size} байт.")


def generate_iv(iv_size: int) -> bytes:
    """
    Создаёт случайный вектор инициализации для режима CBC.
    
    Args:
        iv_size: Размер IV в байтах, должен быть 8.
    
    Returns:
        Сгенерированный IV из 8 байт.
    """
    match iv_size:
        case 8:
            try:
                iv = os.urandom(iv_size)
                if all(b == 0 for b in iv):
                    raise RuntimeError("Сгенерирован нулевой IV")
                return iv
            except NotImplementedError as e:
                raise RuntimeError(f"Система не поддерживает RNG: {e}") from e
            except OSError as e:
                raise RuntimeError(f"Ошибка при генерации IV: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Неожиданная ошибка: {e}") from e
        case _:
            raise ValueError(f"IDEA требует IV 8 байт. Получено: {iv_size} байт.")


def make_padding(data: bytes, block_size_bits: int) -> bytes:
    """
    Дополняет данные до размера, кратного размеру блока, используя ANSI X.923.
    
    Args:
        data: Исходные данные в байтах.
        block_size_bits: Размер блока в битах, должен быть 64.
    
    Returns:
        Данные с добавленным дополнением.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Данные должны быть bytes, получен {type(data).__name__}")
    
    if len(data) == 0:
        raise ValueError("Нельзя добавить дополнение к пустым данным")
    
    match block_size_bits:
        case 64:
            if block_size_bits % 8 != 0:
                raise ValueError(f"Размер блока должен быть кратен 8, получено {block_size_bits}")
            try:
                padder = padding.ANSIX923(block_size_bits).padder()
                data_with_padding = padder.update(data) + padder.finalize()
                return data_with_padding
            except ValueError as e:
                raise ValueError(f"Ошибка дополнения ANSI X.923: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Неожиданная ошибка: {e}") from e
        case _:
            raise ValueError(f"IDEA использует блоки 64 бит. Получено: {block_size_bits} бит.")


def make_unpadding(data_with_padding: bytes, block_size_bits: int) -> bytes:
    """
    Удаляет дополнение из расшифрованных данных.
    
    Args:
        data_with_padding: Данные с дополнением.
        block_size_bits: Размер блока в битах, должен быть 64.
    
    Returns:
        Исходные данные без дополнения.
    """
    if not isinstance(data_with_padding, bytes):
        raise TypeError(f"Данные должны быть bytes, получен {type(data_with_padding).__name__}")
    
    if len(data_with_padding) == 0:
        raise ValueError("Нельзя удалить дополнение из пустых данных")
    
    block_size_bytes = block_size_bits // 8
    
    if len(data_with_padding) % block_size_bytes != 0:
        raise ValueError(f"Данные не выровнены: {len(data_with_padding)} байт, блок {block_size_bytes} байт")
    
    match block_size_bits:
        case 64:
            try:
                unpadder = padding.ANSIX923(block_size_bits).unpadder()
                original_data = unpadder.update(data_with_padding) + unpadder.finalize()
                return original_data
            except ValueError as e:
                raise ValueError(f"Ошибка удаления дополнения: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Неожиданная ошибка: {e}") from e
        case _:
            raise ValueError(f"IDEA использует блоки 64 бит. Получено: {block_size_bits} бит.")


def encrypt_data(key: bytes, iv: bytes, plaintext: bytes, block_size_bits: int) -> bytes:
    """
    Шифрует данные алгоритмом IDEA в режиме CBC.
    
    Args:
        key: Ключ IDEA из 16 байт.
        iv: Вектор инициализации из 8 байт.
        plaintext: Открытый текст для шифрования.
        block_size_bits: Размер блока в битах, должен быть 64.
    
    Returns:
        Зашифрованные данные.
    """
    key_len = len(key) if isinstance(key, bytes) else -1
    iv_len = len(iv) if isinstance(iv, bytes) else -1
    
    match (key_len, iv_len, block_size_bits):
        case (16, 8, 64):
            if not isinstance(plaintext, bytes):
                raise TypeError(f"Текст должен быть bytes, получен {type(plaintext).__name__}")
            
            try:
                cipher = Cipher(algorithms.IDEA(key), modes.CBC(iv))
                encryptor = cipher.encryptor()
                padded_data = make_padding(plaintext, block_size_bits)
                ciphertext = encryptor.update(padded_data) + encryptor.finalize()
                return ciphertext
            except Exception as e:
                raise RuntimeError(f"Ошибка шифрования: {e}") from e
        case (16, 8, _):
            raise ValueError(f"IDEA использует блоки 64 бит. Получено: {block_size_bits} бит.")
        case (16, _, _):
            raise ValueError(f"IV должен быть 8 байт, получено {iv_len}")
        case (_, _, _):
            raise ValueError(f"Ключ должен быть 16 байт, получено {key_len}")


def decrypt_data(key: bytes, iv: bytes, ciphertext: bytes, block_size_bits: int) -> bytes:
    """
    Расшифровывает данные алгоритмом IDEA в режиме CBC.
    
    Args:
        key: Ключ IDEA из 16 байт.
        iv: Вектор инициализации из 8 байт.
        ciphertext: Зашифрованные данные.
        block_size_bits: Размер блока в битах, должен быть 64.
    
    Returns:
        Расшифрованный открытый текст.
    """
    key_len = len(key) if isinstance(key, bytes) else -1
    iv_len = len(iv) if isinstance(iv, bytes) else -1
    
    match (key_len, iv_len, block_size_bits):
        case (16, 8, 64):
            if not isinstance(ciphertext, bytes):
                raise TypeError(f"Шифротекст должен быть bytes, получен {type(ciphertext).__name__}")
            
            if len(ciphertext) == 0:
                raise ValueError("Нельзя расшифровать пустые данные")
            
            block_size_bytes = block_size_bits // 8
            if len(ciphertext) % block_size_bytes != 0:
                raise ValueError(f"Шифротекст повреждён: длина {len(ciphertext)} не кратна {block_size_bytes}")
            
            try:
                cipher = Cipher(algorithms.IDEA(key), modes.CBC(iv))
                decryptor = cipher.decryptor()
                decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
                original_data = make_unpadding(decrypted_padded, block_size_bits)
                return original_data
            except ValueError as e:
                raise ValueError(f"Ошибка расшифрования: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Ошибка расшифрования: {e}") from e
        case (16, 8, _):
            raise ValueError(f"IDEA использует блоки 64 бит. Получено: {block_size_bits} бит.")
        case (16, _, _):
            raise ValueError(f"IV должен быть 8 байт, получено {iv_len}")
        case (_, _, _):
            raise ValueError(f"Ключ должен быть 16 байт, получено {key_len}")