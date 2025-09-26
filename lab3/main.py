import argparse
import const
import os
import sys

from hybrid_crypto_system import CryptoManager


def setup_arg_parser():
    """
    Настраивает парсер аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Гибридная криптосистема Camellia-RSA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Генерация ключей
    gen = subparsers.add_parser("gen", help="Генерация ключей")
    gen.add_argument("--sk", default=const.PATH_TO_SYM_KEY,
                   help="Файл симметричного ключа")
    gen.add_argument("--pub", default=const.PATH_TO_PUBLIC_KEY,
                   help="Файл публичного ключа RSA")
    gen.add_argument("--priv", default=const.PATH_TO_PRIVATE_KEY,
                   help="Файл приватного ключа RSA")
    gen.add_argument("--size", type=int, choices=const.CAMELLIA_KEY_SIZES,
                   default=const.DEFAULT_CAMELLIA_KEY_SIZE,
                   help="Размер ключа Camellia")

    # Шифрование
    enc = subparsers.add_parser("enc", help="Шифрование файла")
    enc.add_argument("input", nargs='?', default=const.PATH_TO_INPUT_FILE,
                   help="Файл для шифрования")
    enc.add_argument("--out", default=const.PATH_TO_ENCRYPTED_FILE,
                   help="Выходной файл")
    enc.add_argument("--sk", default=const.PATH_TO_SYM_KEY,
                   help="Файл симметричного ключа")
    enc.add_argument("--priv", default=const.PATH_TO_PRIVATE_KEY,
                   help="Файл приватного ключа RSA")

    # Дешифрование
    dec = subparsers.add_parser("dec", help="Дешифрование файла")
    dec.add_argument("input", nargs='?', default=const.PATH_TO_ENCRYPTED_FILE,
                   help="Файл для дешифрования")
    dec.add_argument("--out", default=const.PATH_TO_DECRYPTED_FILE,
                   help="Выходной файл")
    dec.add_argument("--sk", default=const.PATH_TO_SYM_KEY,
                   help="Файл симметричного ключа")
    dec.add_argument("--priv", default=const.PATH_TO_PRIVATE_KEY,
                   help="Файл приватного ключа RSA")

    return parser

def execute_command(args):
    """
    Выполняет выбранную команду.
    """
    try:
        match args.command:
            case "gen":
                CryptoManager.generate_keys(args.sk, args.pub, args.priv, args.size)
            case "enc":
                CryptoManager.encrypt_file(args.input, args.out, args.priv, args.sk)
            case "dec":
                CryptoManager.decrypt_file(args.input, args.out, args.priv, args.sk)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Точка входа в программу.
    """
    parser = setup_arg_parser()
    args = parser.parse_args()
    execute_command(args)

if __name__ == '__main__':
    main()