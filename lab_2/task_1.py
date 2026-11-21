import math
import argparse


def parse_args():
    """Обработка аргументов командной строки для шифрования"""
    parser = argparse.ArgumentParser(
        description="Шифрование текста методом постолбцовой транспозиции"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="input.txt",
        help="Файл с исходным текстом",
    )
    parser.add_argument(
        "-k",
        "--key",
        default="key.txt",
        help="Файл с ключом перестановки",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="encrypted.txt",
        help="Файл для зашифрованного текста",
    )
    return parser.parse_args()


def read_file(filename):
    """Чтение текстового файла"""
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def write_file(filename, text):
    """Запись текста в файл"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


def clean_text(text):
    """Удаление пробельных символов из текста"""
    return "".join(text.split())


def encrypt(text, key):
    """Шифрование текста методом столбцовой перестановки"""
    text = clean_text(text)

    columns = len(key)
    rows = math.ceil(len(text) / columns)

    text += "Х" * (rows * columns - len(text))

    matrix = [
        list(text[i * columns:(i + 1) * columns])
        for i in range(rows)
    ]

    key_order = sorted(
        [(symbol, index) for index, symbol in enumerate(key)]
    )

    encrypted_text = ""

    for _, column_index in key_order:
        for row in matrix:
            encrypted_text += row[column_index]

    return encrypted_text


def main():
    """Основная функция программы"""
    args = parse_args()

    text = read_file(args.input)
    key = read_file(args.key).strip()

    encrypted = encrypt(text, key)
    write_file(args.output, encrypted)

    print("Шифрование завершено")


if __name__ == "__main__":
    main()