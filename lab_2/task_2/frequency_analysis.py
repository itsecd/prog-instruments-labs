#!/usr/bin/env python3
"""
Частотный анализ для расшифровки текстов.
Простая версия для новичков.
"""

import os
from collections import Counter

from config import FREQ, CIPHER_FILE


def read_cipher_text(file_path: str) -> str:
    """Читает зашифрованный текст из файла.

    Args:
        file_path (str): Путь к файлу с зашифрованным текстом.

    Returns:
        str: Зашифрованный текст.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def analyze_frequencies(text: str) -> list[tuple[str, int]]:
    """Выполняет частотный анализ текста.

    Args:
        text (str): Текст для анализа.

    Returns:
        list[tuple[str, int]]: Список символов с частотами (от наиболее частых к редким).
    """
    clean_text = text.replace('\n', '').replace(' ', '')
    return Counter(clean_text).most_common()


def print_frequency_analysis(counts: list[tuple[str, int]], total: int) -> None:
    """Выводит таблицу частотного анализа.

    Args:
        counts (list[tuple[str, int]]): Частоты символов.
        total (int): Общее количество символов.
    """
    print("=" * 40)
    print("ЧАСТОТНЫЙ АНАЛИЗ")
    print("=" * 40)
    print("Символ     Кол-во   %       Предл.")
    print("-" * 40)

    for i, (char, count) in enumerate(counts):
        percent = count / total * 100
        suggestion = FREQ[i] if i < len(FREQ) else '?'
        if char == ' ':
            display = '(пробел)'
        elif char == '\n':
            display = '(newline)'
        else:
            display = char
        print(f"{display:<10} {count:<8} {percent:<8.2f} {suggestion}")

    print("-" * 40)


def build_key_from_frequencies(counts: list[tuple[str, int]]) -> dict[str, str]:
    """Строит ключ расшифровки на основе частотного анализа.

    Args:
        counts (list[tuple[str, int]]): Частоты символов (от наиболее частых к редким).

    Returns:
        dict[str, str]: Словарь расшифровки (шифр → оригинал).
    """
    key = {}
    for i, (char, _) in enumerate(counts):
        if i < len(FREQ):
            key[char] = FREQ[i]
    return key


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


def print_decrypted(decrypted: str) -> None:
    """Выводит расшифрованный текст.

    Args:
        decrypted (str): Расшифрованный текст.
    """
    print()
    print("=" * 40)
    print("АВТОМАТИЧЕСКАЯ РАСШИФРОВКА:")
    print("=" * 40)
    print(decrypted)
    print("=" * 40)


def main() -> None:
    """Основная функция."""
    text = read_cipher_text(CIPHER_FILE)
    counts = analyze_frequencies(text)
    total = len(text.replace('\n', '').replace(' ', ''))

    print_frequency_analysis(counts, total)

    key = build_key_from_frequencies(counts)
    decrypted = decrypt_text(text, key)
    print_decrypted(decrypted)


if __name__ == '__main__':
    main()


