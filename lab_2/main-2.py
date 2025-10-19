import argparse

from crypto_operations import FileDecryptor, FileEncryptor
from file_operations import FileManager
from keys import KeyManager


MAX_KEY_SIZE = 448
MIN_KEY_SIZE = 32
DEFAULT_SETTINGS_FILE = "settings.json"
BACKUP_PATH = "/default/backup/"
ENCRYPTION_MODE = "CBC"


def get_valid_key_size():

    while True:
        try:
            key_size = int(input("Введите размер ключа Blowfish (32-448 бит с шагом 8): "))
            if key_size < MIN_KEY_SIZE or key_size > MAX_KEY_SIZE or key_size % 8 != 0:
                print("Ошибка: Размер ключа должен быть 32-448 бит с шагом 8")
                continue
            return key_size
        except ValueError:
            print("Ошибка: Введите целое число")


def main():

    parser = argparse.ArgumentParser(description='Гибридная криптосистема (Blowfish + RSA)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen', '--generate', action='store_true', help='Режим генерации ключей')
    group.add_argument('-enc', '--encrypt', action='store_true', help='Режим шифрования файла')
    group.add_argument('-dec', '--decrypt', action='store_true', help='Режим расшифровки файла')

    parser.add_argument('--key-size', type=int, default=None,
                        help='Размер ключа Blowfish (32-448 бит с шагом 8)')
    parser.add_argument('--settings', type=str, default='settings.json',
                        help='Путь к файлу настроек (по умолчанию: settings.json)')

    args = parser.parse_args()
    settings = FileManager.LoadSettingsFile(args.settings)
    key_manager = KeyManager()

    if args.generate:
        print("\n= Режим генерации ключей =")

        if args.key_size is not None:
            try:
                key_manager.ValidateKeySize(args.key_size)
                key_size = args.key_size
            except ValueError as e:
                print(f"Ошибка размера ключа: {e}")
                return
        else:
            key_size = get_valid_key_size()

        symmetric_key = key_manager.Generate_Symmetric_Key(key_size)
        private_key, public_key = key_manager.generate_asymmetric_keys()

        FileManager.save_keys(
            settings['symmetric_key'],
            settings['public_key'],
            settings['private_key'],
            public_key,
            private_key,
            symmetric_key
        )

        file_manager = FileManager()
        file_manager.backup_keys(settings)

    elif args.encrypt:
        print("\n= Режим шифрования =")
        key_manager.load_private_key(settings['private_key'])
        symmetric_key = key_manager.decrypt_symmetric_key(settings['symmetric_key'])

        encryptor = FileEncryptor(symmetric_key)
        FileManager.encrypt_file(
            settings['initial_file'],
            settings['encrypted_file'],
            encryptor
        )

    elif args.decrypt:
        print("\n= Режим расшифровки =")
        key_manager.load_private_key(settings['private_key'])
        symmetric_key = key_manager.decrypt_symmetric_key(settings['symmetric_key'])

        decryptor = FileDecryptor(symmetric_key)
        FileManager.decrypt_file(
            settings['encrypted_file'],
            settings['decrypted_file'],
            decryptor
        )


if __name__ == "__main__":
    main()