import json
import os
import sys
from typing import Optional, Tuple

from loguru import logger
from config import SIZE

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)
logger.add(
    "polybius.log",
    level="DEBUG",
    rotation="1 MB",
    encoding="utf-8"
)


def read_key_txt(filename: str) -> Tuple[Optional[dict], Optional[str]]:
    """Читает квадрат Полибия из текстового файла и создает маппинг для JSON."""
    logger.debug(f"Попытка чтения ключа из файла: {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"Ошибка: файл {filename}! не найден")
        raise FileNotFoundError(f"Ошибка: файл {filename} не найден!")

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    square = []
    for line in lines[1:]:
        line = line.rstrip('\n')
        if line:
            chars = line[4:].split()
            logger.debug(f"Чтение строки ключа: '{line}' -> Элементы: {chars}")
            square.append(chars)

    if len(square) != SIZE:
        logger.error(f"Ошибка: в файле {len(square)} строк, ожидалось {SIZE}")
        return None, None

    encrypt_map = {}
    alphabet = ""

    for i in range(SIZE):
        if len(square[i]) != SIZE:
            logger.error(
                f"Ошибка: в строке {i+1} {len(square[i])} элементов, "
                f"ожидалось {SIZE}"
            )
            return None, None

        for j in range(SIZE):
            char = square[i][j]
            if char != '*':
                if char == '_':
                    encrypt_map[' '] = f"{i+1:01d}{j+1:01d}"
                    alphabet += ' '
                else:
                    encrypt_map[char] = f"{i+1:01d}{j+1:01d}"
                    alphabet += char

    logger.info(f"Ключ успешно загружен! Размер: {len(encrypt_map)} символов.")
    return encrypt_map, alphabet


def read_original_text(filename: str) -> str:
    """Читает исходный текст из файла."""
    logger.debug(f"Попытка чтения исходного текста из файла: {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"Ошибка: файл {filename} не найден!")
        raise FileNotFoundError(f"Ошибка: файл {filename} не найден!")

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        
    logger.info(f"Исходный текст прочитан. Длина: {len(text)} символов.")
    return text.upper()


def save_key_json(filename: str, encrypt_map: dict) -> None:
    """Сохраняет маппинг в JSON формате."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(encrypt_map, f, ensure_ascii=False, indent=2)
    logger.debug(f"Ключ сохранен в {filename}")


def encrypt_polybius(text: str, encrypt_map: dict) -> str:
    """Шифрует текст, заменяя символы на координаты."""
    logger.debug("Начало процесса шифрования...")
    encrypted_pairs = []
    unknown_chars = set()

    for char in text:
        if char in encrypt_map:
            encrypted_pairs.append(encrypt_map[char])
        else:
            unknown_chars.add(char)
            encrypted_pairs.append('??')

    if unknown_chars:
        logger.warning(
            f"Следующие символы не найдены в ключе: {sorted(unknown_chars)}"
        )

    encrypted_text = ' '.join(encrypted_pairs)
    logger.info("Шифрование успешно завершено.")
    return encrypted_text


def decrypt_polybius(encrypted_text: str, encrypt_map: dict) -> str:
    """Дешифрует текст из координат обратно в буквы."""
    logger.debug("Начало процесса дешифрования...")
    decrypt_map = {v: k for k, v in encrypt_map.items()}
    pairs = encrypted_text.split(' ')
    decrypted_text = []
    unknown_pairs = set()

    for pair in pairs:
        if pair in decrypt_map:
            decrypted_text.append(decrypt_map[pair])
        else:
            unknown_pairs.add(pair)
            decrypted_text.append('?')

    if unknown_pairs:
        logger.warning(
            f"Следующие координаты не найдены в ключе: {sorted(unknown_pairs)}"
        )

    logger.info("Дешифрование успешно завершено.")
    return ''.join(decrypted_text)


def save_to_file(filename: str, content: str) -> None:
    """Сохраняет содержимое в файл."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Результат успешно сохранен в файл: {filename}")


def print_key_preview(filename: str) -> None:
    """Выводит содержимое key.txt для наглядности."""
    logger.debug("Вывод превью ключа в лог")
    with open(filename, 'r', encoding='utf-8') as f:
        logger.info(f"Используемый ключ шифрования (key.txt):\n{f.read()}")


def main() -> None:
    """Основная функция программы."""
    try:
        logger.info("--- Запуск программы шифрования ---")

        encrypt_map, alphabet = read_key_txt('key.txt')
        if encrypt_map is None:
            logger.critical("Ошибка при чтении key.txt. Проверьте формат файла.")
            sys.exit(1)

        logger.debug(f"Алфавит из ключа: {alphabet}")
        logger.info("Создание key.json на основе key.txt.")
        save_key_json('key.json', encrypt_map)
        print_key_preview('key.txt')

        original_text = read_original_text('original.txt')
        logger.debug(f"Исходный текст (original.txt):\n{original_text}")

        encrypted_text = encrypt_polybius(original_text, encrypt_map)
        decrypted_text = decrypt_polybius(encrypted_text, encrypt_map)

        save_to_file('encrypted.txt', encrypted_text)

        logger.debug(f"Зашифрованный текст (encrypted.txt):\n{encrypted_text}")
        logger.debug(f"Всего пар координат: {len(encrypted_text.split())}")

        logger.debug(f"Расшифрованный текст (проверка):\n{decrypted_text}")
        logger.debug(f"Всего символов: {len(decrypted_text)}")

        if original_text == decrypted_text:
            logger.success("Шифрование работает корректно. Тексты совпадают.")
        else:
            logger.error("Расшифрованный текст не совпадает с оригиналом!")

        logger.info("--- Программа успешно завершена ---")

    except FileNotFoundError as e:
        logger.exception("Критическая ошибка при работе с файлами")
        sys.exit(1)
    except Exception as e:
        logger.exception("Непредвиденная ошибка в процессе работы")
        sys.exit(1)


if __name__ == "__main__":
    main()
