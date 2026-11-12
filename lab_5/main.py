from key_gen import generate_keys
from encryption import encrypt
from decryption import decrypt
from typing import Literal


def main(settings: dict[str, str | int | bytes], mode: Literal["gen", "enc", "dec"]) -> None:
    """
    Точка входа в логику программы
    Args:
        settings (dict): Параметры приложения
        mode (str): Режим работы программы
    """
    match mode:
        case "gen":
            generate_keys(settings)
        case "enc":
            encrypt(settings)
        case "dec":
            decrypt(settings)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--settings", help="Путь до файла с параметрами", default="./settings.json")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen', '--generation', action='store_const', const='gen', dest='mode', help='Запускает режим генерации ключей')
    group.add_argument('-enc', '--encryption', action='store_const', const='enc', dest='mode', help='Запускает режим шифрования')
    group.add_argument('-dec', '--decryption', action='store_const', const='dec', dest='mode', help='Запускает режим дешифрования')

    args = parser.parse_args()

    try:
        with open(args.settings, mode="r", encoding="utf-8") as input:
            settings = json.load(input)
        settings.setdefault("rsa_public_exponent", 65537)
        settings.setdefault("rsa_key_size", 2048)
        settings.setdefault("3des_key_size", 64)
        # JSON не поддерживает хранения байтовых объектов. Можно использовать кодировку в base64, но массив чисел выглядит приятнее. 
        settings.setdefault("3des_init_vector", (255, 241, 68, 43, 13, 31, 35, 86))
        settings["3des_init_vector"] = bytes(settings["3des_init_vector"])

        main(settings, args.mode)
    except FileNotFoundError as e:
        print(f"Файл {e.filename} не найден!")
    except Exception as e:
        print(e)
