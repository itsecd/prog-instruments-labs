import json
import sys


def load_key(key_file):
    """Загружает ключ из JSON файла."""
    try:
        with open(key_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл {key_file} не найден")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Ошибка: Файл {key_file} содержит некорректный JSON")
        sys.exit(1)


def decrypt_text(encrypted_text, key):
    """Дешифрует текст с использованием ключа."""
    decrypted = []

    for char in encrypted_text:
        if char in key:
            decrypted.append(key[char])
        else:
            decrypted.append(char)

    return "".join(decrypted)


def main():
    """Основная функция программы."""
    key_file = "key.json"
    input_file = "original.txt"
    output_file = "decrypted.txt"

    print(f"Загружаем ключ из {key_file}.")
    key = load_key(key_file)
    print(f"Загружено {len(key)} замен")

    print(f"Читаем зашифрованный текст из {input_file}.")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            encrypted_text = f.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_file} не найден")
        sys.exit(1)

    print("Дешифруем текст.")
    decrypted_text = decrypt_text(encrypted_text, key)

    print(f"Сохраняем результат в {output_file}.")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(decrypted_text)

    print("\nРасшифрованный текст:")
    print(decrypted_text)
    print(f"\nТекст сохранен в {output_file}")


if __name__ == "__main__":
    main()
