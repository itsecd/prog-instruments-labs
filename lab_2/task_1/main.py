#!/usr/bin/env python3
"""
Консольное меню для шифрования и дешифрования текстов.
"""

import os
import sys

from constants import DEFAULT_KEY_FILE, DEFAULT_TEXT_FILE, RESULTS_DIR


def ensure_results_dir():
    """Создаёт директорию для результатов шифрования/дешифрования.

    Проверяет существование директории RESULTS_DIR и создаёт её,
    если она отсутствует. Используется для хранения выходных файлов.

    Side Effects:
        Создаёт директорию 'results' в текущей рабочей директории,
        если она не существует.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def read_key(key_file: str) -> dict[str, str]:
    """Читает файл с ключом и создаёт словарь подстановки.

    Формат ключа: пары символов (оригинал + замена) без разделителей.
    Пример: 'а1б2' → {'а': '1', 'б': '2'}.
    Поддерживает как строчные, так и заглавные русские буквы.

    Args:
        key_file (str): Путь к файлу с ключом шифрования.

    Returns:
        dict[str, str]: Словарь замены символов.

    Raises:
        FileNotFoundError: Если файл с ключом не найден.
    """
    with open(key_file, 'r', encoding='utf-8') as f:
        key = f.read().strip()

    substitution = {}
    for i in range(0, len(key) - 1, 2):
        original = key[i]
        replacement = key[i + 1]
        substitution[original] = replacement
    return substitution


def invert_key(substitution: dict[str, str]) -> dict[str, str]:
    """Инвертирует словарь подстановки для дешифрования.

    Меняет ключи и значения местами: оригинальные значения становятся
    ключами, а ключи — значениями. Используется для обратного
    преобразования зашифрованного текста.

    Args:
        substitution (dict[str, str]): Прямой словарь замены для шифрования.

    Returns:
        dict[str, str]: Инвертированный словарь для дешифрования.
    """
    inverted = {}
    for k, v in substitution.items():
        inverted[v] = k
    return inverted


def encrypt_text(text: str, substitution: dict[str, str]) -> str:
    """Выполняет подстановку символов в тексте по словарю замены.

    Проходит по каждому символу текста и заменяет его согласно словарю.
    Символы, отсутствующие в словаре, остаются без изменений.

    Args:
        text (str): Исходный текст для обработки.
        substitution (dict[str, str]): Словарь замены символов.

    Returns:
        str: Обработанный текст с применённой подстановкой.
    """
    result = []
    for char in text:
        if char in substitution:
            result.append(substitution[char])
        else:
            result.append(char)
    return ''.join(result)


def process_file(key_file: str, input_file: str, output_file: str, decrypt: bool = False):
    """Обрабатывает файл: шифрует или дешифрует содержимое.

    Читает ключ подстановки, при необходимости инвертирует его для
    дешифрования, применяет замену к содержимому входного файла и
    записывает результат в выходной файл.

    Args:
        key_file (str): Путь к файлу с ключом шифрования.
        input_file (str): Путь к входному файлу с текстом.
        output_file (str): Путь к выходному файлу для результата.
        decrypt (bool): Флаг дешифрования (True) или шифрования (False).

    Side Effects:
        Создаёт или перезаписывает файл по пути output_file.
        Выводит сообщение об успешном завершении в консоль.

    Raises:
        FileNotFoundError: Если входной файл или файл ключа не найдены.
    """
    # Чтение ключа
    substitution = read_key(key_file)
    
    # Инвертирование ключа для дешифрования
    if decrypt:
        substitution = invert_key(substitution)
    
    # Чтение текста
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Обработка
    result = encrypt_text(text, substitution)
    
    # Запись результата
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"✓ Результат сохранён в {output_file}")


def get_input(prompt: str, default: str = None) -> str:
    """Запрашивает ввод имени файла у пользователя.

    Выводит приглашение с поддержкой значения по умолчанию.
    Проверяет, что введённое имя имеет расширение .txt.
    При некорректном вводе запрашивает повторно (цикл).

    Args:
        prompt (str): Текст приглашения для пользователя.
        default (str, optional): Значение по умолчанию.

    Returns:
        str: Введённое пользователем имя файла или значение по умолчанию.

    Side Effects:
        Выводит сообщение об ошибке при некорректном вводе.
    """
    display_prompt = f"{prompt} (по умолчанию: {default})" if default else prompt
    while True:
        value = input(f"Введите {display_prompt} > ").strip()
        if not value:
            return default
        if not value.endswith('.txt'):
            print("✗ Ошибка: имя файла должно иметь расширение .txt")
            continue
        return value


def menu_encrypt():
    """Запускает интерактивное меню шифрования текста.

    Запрашивает у пользователя пути к файлу ключа, входному файлу
    и файлу результата. Вызывает process_file для шифрования.

    Side Effects:
        Выводит приглашения для ввода имён файлов.
        Создаёт директорию результатов при необходимости.
        Выводит сообщения об ошибках при неудаче.
    """
    print("\n=== Шифрование текста ===")
    key_file = get_input("Файл с ключом", DEFAULT_KEY_FILE)
    input_file = get_input("Файл с текстом", DEFAULT_TEXT_FILE)
    output_file = get_input("Файл для результата", RESULTS_DIR + "/encrypted.txt")

    ensure_results_dir()

    try:
        process_file(key_file, input_file, output_file, decrypt=False)
    except FileNotFoundError as e:
        print(f"✗ Ошибка: файл не найден - {e}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")


def menu_decrypt():
    """Запускает интерактивное меню дешифрования текста.

    Запрашивает у пользователя пути к файлу ключа, зашифрованному
    файлу и файлу результата. Вызывает process_file для дешифрования.

    Side Effects:
        Выводит приглашения для ввода имён файлов.
        Создаёт директорию результатов при необходимости.
        Выводит сообщения об ошибках при неудаче.
    """
    print("\n=== Дешифрование текста ===")
    key_file = get_input("Файл с ключом", DEFAULT_KEY_FILE)
    input_file = get_input("Файл с зашифрованным текстом", RESULTS_DIR +"/encrypted.txt")
    output_file = get_input("Файл для результата", RESULTS_DIR + "/decrypted.txt")

    ensure_results_dir()

    try:
        process_file(key_file, input_file, output_file, decrypt=True)
    except FileNotFoundError as e:
        print(f"✗ Ошибка: файл не найден - {e}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")


def main():
    """Запускает главное меню программы шифрования/дешифрования.

    Отображает меню с выбором операций: шифрование, дешифрование
    или выход из программы. Работает в цикле до выбора пользователем
    пункта "Выход".

    Side Effects:
        Выводит главное меню и приглашения для выбора.
        Вызывает функции menu_encrypt() или menu_decrypt()
        в зависимости от выбора пользователя.
    """
    while True:
        print("\n" + "=" * 35)
        print("         МЕНЮ ПРОГРАММЫ")
        print("=" * 35)
        print("1. Зашифровать текст")
        print("2. Дешифровать текст")
        print("0. Выход")
        print("=" * 35)

        choice = input("Выберите пункт меню: ").strip()

        if choice == '1':
            menu_encrypt()
        elif choice == '2':
            menu_decrypt()
        elif choice == '0':
            print("Выход из программы. До свидания!")
            break
        else:
            print("✗ Неверный выбор. Попробуйте снова.")


if __name__ == '__main__':
    main()
