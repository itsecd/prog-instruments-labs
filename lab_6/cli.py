import argparse
from config_utils import load_config
from crypto_utils import generate_keys, encrypt_file, decrypt_file


def main():
    parser = argparse.ArgumentParser(description="Гибридная криптосистема RSA + AES")
    parser.add_argument("-c", "--config", default="config.json", help="путь к json-файлу настроек")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-gen", "--generation", action="store_true", help="генерация ключей")
    group.add_argument("-enc", "--encryption", action="store_true", help="шифрование файла")
    group.add_argument("-dec", "--decryption", action="store_true", help="дешифрование файла")

    args = parser.parse_args()
    settings = load_config(args.config)

    match True:
        case _ if args.generation:
            print("[1/4] Генерирую AES ключ...")
            print("[2/4] Генерирую RSA ключи...")
            generate_keys(
                settings["encrypted_key_file"],
                settings["public_key_file"],
                settings["private_key_file"],
                int(settings["aes_key_size"])
            )
            print("[3/4] Сохраняю RSA ключи...")
            print("[4/4] Сохраняю зашифрованный AES ключ...")
            print("Готово: ключи созданы")

        case _ if args.encryption:
            print("[1/3] Загружаю открытый RSA ключ...")
            print("[2/3] Шифрую файл алгоритмом AES-CBC...")
            encrypt_file(
                settings["input_file"],
                settings["public_key_file"],
                settings["encrypted_key_file"],
                settings["encrypted_file"],
                int(settings["aes_key_size"])
            )
            print("[3/3] Сохраняю зашифрованный файл...")
            print("Готово: файл зашифрован")

        case _ if args.decryption:
            print("[1/3] Загружаю закрытый RSA ключ и зашифрованный AES ключ...")
            print("[2/3] Расшифровываю файл алгоритмом AES-CBC...")
            decrypt_file(
                settings["encrypted_file"],
                settings["private_key_file"],
                settings["encrypted_key_file"],
                settings["decrypted_file"]
            )
            print("[3/3] Сохраняю расшифрованный файл...")
            print("Готово: файл расшифрован")


if __name__ == "__main__":
    main()