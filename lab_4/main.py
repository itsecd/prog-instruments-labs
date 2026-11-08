import argparse

from asymmetric_crypto import (
    decrypt_symmetric_key,
    encrypt_symmetric_key,
    generate_rsa_key,
    load_private_key,
    save_private_key,
    save_public_key,
)
from file_utils import read_binary_file, read_json_file, write_binary_file, write_json_file
from symmetric_crypto import (
    decrypt_file_camellia,
    encrypt_file_camellia,
    generate_symmetric_key,
)


def request_settings(settings: dict) -> dict:
    """
    Запрашивает у пользователя пути к ключам и обновляет настройки.

    :param settings: исходный словарь настроек
    :return: обновлённый словарь настроек
    """
    private = input(f"Путь к приватному ключу [{settings['private_key']}]: ")
    if private:
        settings["private_key"] = private

    public = input(f"Путь к публичному ключу [{settings['public_key']}]: ")
    if public:
        settings["public_key"] = public

    symmetric = input(f"Путь к симметричному ключу [{settings['symmetric_key']}]: ")
    if symmetric:
        settings["symmetric_key"] = symmetric

    return settings


def generation_mode(settings: dict) -> None:
    """
    Генерирует ключи гибридной криптосистемы.

    :param settings: словарь с настройками
    :return: None
    """
    print("Генерация ключа Camellia")
    symmetric_key = generate_symmetric_key(settings["key_size"])

    print("Генерация ключей RSA")
    private_key, public_key = generate_rsa_key()

    print("Сохранение закрытого ключа RSA")
    save_private_key(private_key, settings["private_key"])

    print("Сохранение открытого ключа RSA")
    save_public_key(public_key, settings["public_key"])

    print("Шифрование ключа Camellia открытым ключом RSA")
    encrypted_key = encrypt_symmetric_key(symmetric_key, public_key)

    print("Сохранение зашифрованного ключа Camellia")
    write_binary_file(settings["symmetric_key"], encrypted_key)

    print("Генерация завершена")


def encryption_mode(settings: dict) -> None:
    """
    Шифрует файл гибридной криптосистемой.

    :param settings: словарь с настройками
    :return: None
    """
    print("Загрузка закрытого ключа RSA")
    private_key = load_private_key(settings["private_key"])

    print("Загрузка зашифрованного ключа Camellia")
    encrypted_key = read_binary_file(settings["symmetric_key"])

    print("Расшифрование ключа Camellia")
    symmetric_key = decrypt_symmetric_key(encrypted_key, private_key)

    print("Шифрование файла алгоритмом Camellia")
    encrypt_file_camellia(
        settings["input_file"],
        settings["encrypted_file"],
        symmetric_key,
    )

    print("Шифрование завершено")


def decryption_mode(settings: dict) -> None:
    """
    Дешифрует файл гибридной криптосистемой.

    :param settings: словарь с настройками
    :return: None
    """
    print("Загрузка закрытого ключа RSA")
    private_key = load_private_key(settings["private_key"])

    print("Загрузка зашифрованного ключа Camellia")
    encrypted_key = read_binary_file(settings["symmetric_key"])

    print("Расшифрование ключа Camellia")
    symmetric_key = decrypt_symmetric_key(encrypted_key, private_key)

    print("Дешифрование файла алгоритмом Camellia")
    decrypt_file_camellia(
        settings["encrypted_file"],
        settings["decrypted_file"],
        symmetric_key,
    )

    print("Дешифрование завершено")


def main() -> None:
    """
    Запускает выбранный режим программы.
    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gen", "--generation", dest="generation")
    group.add_argument("--enc", "--encryption", dest="encryption")
    group.add_argument("--dec", "--decryption", dest="decryption")

    parser.add_argument("--private-key", dest="private_key")
    parser.add_argument("--public-key", dest="public_key")
    parser.add_argument("--symmetric-key", dest="symmetric_key")
    parser.add_argument("--key-size", dest="key_size", type=int, choices=(128, 192, 256))

    args = parser.parse_args()

    try:
        mode = (
            args.generation is not None,
            args.encryption is not None,
            args.decryption is not None,
        )
        settings_path = args.generation or args.encryption or args.decryption
        settings = read_json_file(settings_path)

        settings = request_settings(settings)
        write_json_file(settings_path, settings)

        if args.private_key is not None:
            settings["private_key"] = args.private_key
        if args.public_key is not None:
            settings["public_key"] = args.public_key
        if args.symmetric_key is not None:
            settings["symmetric_key"] = args.symmetric_key
        if args.key_size is not None:
            settings["key_size"] = args.key_size

        match mode:
            case (True, False, False):
                print("Режим генерации ключей")
                generation_mode(settings)
            case (False, True, False):
                print("Режим шифрования")
                encryption_mode(settings)
            case (False, False, True):
                print("Режим дешифрования")
                decryption_mode(settings)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()
