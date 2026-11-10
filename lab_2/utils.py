"""Вспомогательные функции"""

from cryptography.hazmat.primitives import padding


def read_binary_file(filepath: str) -> bytes:
    """Чтение бинарного файла"""
    with open(filepath, 'rb') as f:
        return f.read()


def write_binary_file(filepath: str, data: bytes) -> None:
    """Запись бинарного файла"""
    with open(filepath, 'wb') as f:
        f.write(data)


def read_text_file(filepath: str) -> bytes:
    """Чтение текстового файла в байты"""
    with open(filepath, 'rb') as f:
        return f.read()


def write_text_file(filepath: str, data: bytes) -> None:
    """Запись текстового файла из байт"""
    with open(filepath, 'wb') as f:
        f.write(data)


def pad_data(data: bytes, block_size: int) -> bytes:
    """Дополнение данных до размера блока
    
    Args:
        data: Данные для дополнения
        block_size: Размер блока в байтах
    """
    padder = padding.ANSIX923(block_size * 8).padder()
    return padder.update(data) + padder.finalize()


def unpad_data(data: bytes, block_size: int) -> bytes:
    """Удаление дополнения из данных
    
    Args:
        data: Данные с дополнением
        block_size: Размер блока в байтах
    """
    unpadder = padding.ANSIX923(block_size * 8).unpadder()
    return unpadder.update(data) + unpadder.finalize()