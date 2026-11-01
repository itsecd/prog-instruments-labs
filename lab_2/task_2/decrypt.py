#!/usr/bin/env python3
"""
Расшифровка текста с известным ключом.
"""

import os

from config import KEY_FILE, CIPHER_FILE, RESULTS_DIR


def read_key(key_file: str) -> dict[str, str]:
    """Читает файл с ключом и создаёт словарь подстановки.

    Формат ключа: пары символов (символ шифра + расшифровка) без разделителей.
    Пример: '!р$ю' → {'!': 'р', '$': 'ю'}.

    Args:
        key_file (str): Путь к файлу с ключом.

    Returns:
        dict[str, str]: Словарь расшифровки (шифр → оригинал).

    Raises:
        FileNotFoundError: Если файл с ключом не найден.
    """
    with open(key_file, 'r', encoding='utf-8') as f:
        key = f.read().strip()

    substitution = {}
    for i in range(0, len(key) - 1, 2):
        cipher = key[i]
        plain = key[i + 1]
        substitution[cipher] = plain
    return substitution

def decrypt_text(text: str, key: dict[str, str]) -> str:
    """Расшифровывает текст с использованием ключа.

    Args:
        text (str): Зашифрованный текст.
        key (dict[str, str]): Словарь расшифровки.

    Returns:
        str: Расшифрованный текст.
    """
    decrypted = ""
    for c in text:
        if c in key:
            decrypted += key[c]
        else:
            decrypted += c
    return decrypted


def main() -> None:
    """Основная функция."""
    key = read_key(KEY_FILE)

    # Читаем зашифрованный текст
    with open(CIPHER_FILE, 'r', encoding='utf-8') as f:
        encrypted = f.read()

    # Расшифровываем
    decrypted = decrypt_text(encrypted, key)

    # Выводим результат
    print("=" * 40)
    print("РАСШИФРОВАННЫЙ ТЕКСТ:")
    print("=" * 40)
    print(decrypted)
    print("=" * 40)

    # Создаём папку results
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Сохраняем в папку results
    with open(RESULTS_DIR + '/decrypted.txt', 'w', encoding='utf-8') as f:
        f.write(decrypted)

    print()
    print(f"Сохранено в {RESULTS_DIR}/decrypted.txt")


if __name__ == '__main__':
    main()
