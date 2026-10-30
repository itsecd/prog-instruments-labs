import json
import os
import sys

from config import SIZE


def read_key_txt(filename):
    """
    Читает квадрат Полибия из текстового файла и создает маппинг для JSON.

    Args:
        filename (str): Путь к файлу с ключом

    Returns:
        tuple: (encrypt_map, alphabet)

    Raises:
        FileNotFoundError: Если файл не найден
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Ошибка: файл {filename} не найден!")

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    square = []
    for line in lines[1:]:
        line = line.rstrip("\n")
        if line:
            chars = line[4:].split("   ")
            square.append(chars)

    if len(square) != SIZE:
        print(f"Ошибка: в файле {len(square)} строк, ожидалось {SIZE}")
        return None, None

    encrypt_map = {}
    alphabet = ""

    for i in range(SIZE):
        if len(square[i]) != SIZE:
            print(
                f"Ошибка: в строке {i + 1} {len(square[i])} элементов, ожидалось {SIZE}"
            )
            return None, None

        for j in range(SIZE):
            char = square[i][j]
            if char != "*":
                if char == "_":
                    encrypt_map[" "] = f"{i + 1:01d}{j + 1:01d}"
                    alphabet += " "
                else:
                    encrypt_map[char] = f"{i + 1:01d}{j + 1:01d}"
                    alphabet += char

    return encrypt_map, alphabet


def read_original_text(filename):
    """
    Читает исходный текст из файла.

    Args:
        filename (str): Путь к файлу с исходным текстом

    Returns:
        str: Прочитанный текст в верхнем регистре

    Raises:
        FileNotFoundError: Если файл не найден
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Ошибка: файл {filename} не найден!")

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    return text.upper()


def save_key_json(filename, encrypt_map):
    """Сохраняет маппинг в JSON формате."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(encrypt_map, f, ensure_ascii=False, indent=2)


def encrypt_polybius(text, encrypt_map):
    """Шифрует текст, заменяя символы на координаты."""
    encrypted_pairs = []
    unknown_chars = set()

    for char in text:
        if char in encrypt_map:
            encrypted_pairs.append(encrypt_map[char])
        else:
            unknown_chars.add(char)
            encrypted_pairs.append("??")

    if unknown_chars:
        print(
            "Предупреждение: следующие символы не найдены в ключе: "
            f"{sorted(unknown_chars)}"
        )

    return " ".join(encrypted_pairs)


def decrypt_polybius(encrypted_text, encrypt_map):
    """Дешифрует текст из координат обратно в буквы."""
    decrypt_map = {v: k for k, v in encrypt_map.items()}

    pairs = encrypted_text.split(" ")
    decrypted_text = []
    unknown_pairs = set()

    for pair in pairs:
        if pair in decrypt_map:
            decrypted_text.append(decrypt_map[pair])
        else:
            unknown_pairs.add(pair)
            decrypted_text.append("?")

    if unknown_pairs:
        print(f"Следующие координаты не найдены в ключе: {sorted(unknown_pairs)}")

    return "".join(decrypted_text)


def save_to_file(filename, content):
    """Сохраняет содержимое в файл."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def print_key_preview(filename):
    """Выводит содержимое key.txt для наглядности."""
    print("\nИспользуемый ключ шифрования (key.txt):")
    with open(filename, "r", encoding="utf-8") as f:
        print(f.read())


def main():
    """Основная функция программы."""
    try:
        print("\nЧтение ключа из key.txt.")
        encrypt_map, alphabet = read_key_txt("key.txt")

        if encrypt_map is None:
            print("Ошибка при чтении key.txt. Проверьте формат файла.")
            sys.exit(1)

        print("Ключ успешно загружен!")
        print(f"Найдено символов в ключе: {len(encrypt_map)}")
        print(f"Алфавит из ключа: {alphabet}")

        print("Создание key.json на основе key.txt.")
        save_key_json("key.json", encrypt_map)

        print_key_preview("key.txt")

        print("\nЧтение исходного текста из original.txt")
        original_text = read_original_text("original.txt")

        print("Исходный текст (original.txt):")
        print(original_text)
        print(f"Длина исходного текста: {len(original_text)} символов\n")

        encrypted_text = encrypt_polybius(original_text, encrypt_map)

        decrypted_text = decrypt_polybius(encrypted_text, encrypt_map)

        save_to_file("encrypted.txt", encrypted_text)

        print("Зашифрованный текст (encrypted.txt):")
        print(encrypted_text)
        print(f"\nВсего пар координат: {len(encrypted_text.split())}\n")

        print("Расшифрованный текст (проверка):")
        print(decrypted_text)
        print(f"Всего символов: {len(decrypted_text)}\n")

        if original_text == decrypted_text:
            print("Шифрование работает корректно.")
        else:
            print("Расшифрованный текст не совпадает с оригиналом.")

    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nНепредвиденная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
