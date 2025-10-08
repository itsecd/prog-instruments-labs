import argparse
import json
import os
from config import Config
from asymmetric_crypto import AsymmetricCrypto
from symmetric_crypto import SymmetricCrypto
from file_manager import FileManager


class HybridCryptoSystem:
    """Гибридная криптосистема, объединяющая симметричное и асимметричное шифрование"""

    def __init__(self, config: Config):
        self.config = config
        self.asymmetric = AsymmetricCrypto()
        self.symmetric = SymmetricCrypto()
        self.files = FileManager()

    def generate_and_save_keys(self, key_size: int = 256) -> None:
        """Генерирует и сохраняет все необходимые ключи"""
        private_key, public_key = self.asymmetric.generate_keys()
        symmetric_key = self.symmetric.generate_key(key_size)

        self.files.save_key(public_key, self.config.PATHS['PUBLIC_KEY'])
        self.files.save_key(private_key, self.config.PATHS['SECRET_KEY'])

        encrypted_sym_key = self.asymmetric.encrypt_with_public_key(
            public_key, symmetric_key
        )
        self.files.save_file(
            self.config.PATHS['SYMMETRIC_KEY'], encrypted_sym_key
        )

        print("Ключи успешно сгенерированы и сохранены.")

    def encrypt_file(self, input_file: str = None, output_file: str = None) -> None:
        """Шифрует файл с использованием гибридной системы"""
        input_path = input_file or self.config.PATHS['INITIAL_FILE']
        output_path = output_file or self.config.PATHS['ENCRYPTED_FILE']

        private_key = self.files.load_private_key(self.config.PATHS['SECRET_KEY'])
        encrypted_sym_key = self.files.load_file(self.config.PATHS['SYMMETRIC_KEY'])
        symmetric_key = self.asymmetric.decrypt_with_private_key(
            private_key, encrypted_sym_key
        )

        data = self.files.load_file(input_path)
        encrypted_data = self.symmetric.encrypt_data(data, symmetric_key)

        self.files.save_file(output_path, encrypted_data)
        print(f"Файл успешно зашифрован и сохранён в {output_path}")

    def decrypt_file(self, input_file: str = None, output_file: str = None) -> None:
        """Дешифрует файл с использованием гибридной системы"""
        input_path = input_file or self.config.PATHS['ENCRYPTED_FILE']
        output_path = output_file or self.config.PATHS['DECRYPTED_FILE']

        private_key = self.files.load_private_key(self.config.PATHS['SECRET_KEY'])
        encrypted_sym_key = self.files.load_file(self.config.PATHS['SYMMETRIC_KEY'])
        symmetric_key = self.asymmetric.decrypt_with_private_key(
            private_key, encrypted_sym_key
        )

        encrypted_data = self.files.load_file(input_path)
        decrypted_data = self.symmetric.decrypt_data(encrypted_data, symmetric_key)

        self.files.save_file(output_path, decrypted_data)
        print(f"Файл успешно расшифрован и сохранён в {output_path}")

    def encrypt_with_custom_keys(
            self,
            input_file: str,
            output_file: str,
            public_key_path: str,
            key_size: int = 256
    ) -> None:
        """Шифрует файл с использованием пользовательских ключей"""
        symmetric_key = self.symmetric.generate_key(key_size)
        public_key = self.files.load_public_key(public_key_path)
        encrypted_sym_key = self.asymmetric.encrypt_with_public_key(
            public_key, symmetric_key
        )

        data = self.files.load_file(input_file)
        encrypted_data = self.symmetric.encrypt_data(data, symmetric_key)

        self.files.save_file(output_file, encrypted_sym_key + encrypted_data)
        print(
            f"Файл успешно зашифрован с пользовательскими ключами "
            f"и сохранён в {output_file}"
        )

    def decrypt_with_custom_keys(
            self,
            input_file: str,
            output_file: str,
            private_key_path: str
    ) -> None:
        """Дешифрует файл с использованием пользовательских ключей"""
        encrypted_data = self.files.load_file(input_file)
        encrypted_sym_key = encrypted_data[:256]
        encrypted_content = encrypted_data[256:]

        private_key = self.files.load_private_key(private_key_path)
        symmetric_key = self.asymmetric.decrypt_with_private_key(
            private_key, encrypted_sym_key
        )

        decrypted_data = self.symmetric.decrypt_data(encrypted_content, symmetric_key)

        self.files.save_file(output_file, decrypted_data)
        print(
            f"Файл успешно расшифрован с пользовательскими ключами "
            f"и сохранён в {output_file}"
        )


def create_default_config():
    """Создает файл настроек по умолчанию"""
    config = {
        "paths": {
            "PUBLIC_KEY": "public_key.pem",
            "SECRET_KEY": "private_key.pem",
            "SYMMETRIC_KEY": "symmetric_key.bin",
            "INITIAL_FILE": "original.txt",
            "ENCRYPTED_FILE": "encrypted.bin",
            "DECRYPTED_FILE": "decrypted.txt"
        },
        "settings": {
            "default_key_size": 256,
            "encryption_algorithm": "RSA+AES"
        }
    }

    with open('default_settings.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("Создан файл настроек по умолчанию: default_settings.json")


def main():
    """Основная функция для работы с командной строкой"""
    parser = argparse.ArgumentParser(
        description="Гибридная криптосистема (RSA + AES)"
    )
    parser.add_argument(
        '-s', '--settings', required=True,
        help='Путь к файлу настроек settings.json'
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    gen_parser = subparsers.add_parser('gen', help='Генерация ключей')
    gen_parser.add_argument(
        '--key-size', type=int, default=256,
        choices=[128, 192, 256],
        help='Размер симметричного ключа'
    )

    enc_parser = subparsers.add_parser('enc', help='Шифрование файла')
    enc_parser.add_argument('--input', help='Путь к исходному файлу')
    enc_parser.add_argument('--output', help='Путь для сохранения зашифрованного файла')

    dec_parser = subparsers.add_parser('dec', help='Дешифрование файла')
    dec_parser.add_argument('--input', help='Путь к зашифрованному файлу')
    dec_parser.add_argument('--output', help='Путь для сохранения дешифрованного файла')

    custom_enc_parser = subparsers.add_parser(
        'custom-enc', help='Шифрование с пользовательскими ключами'
    )
    custom_enc_parser.add_argument(
        '--input', required=True, help='Путь к исходному файлу'
    )
    custom_enc_parser.add_argument(
        '--output', required=True, help='Путь для сохранения зашифрованного файла'
    )
    custom_enc_parser.add_argument(
        '--pub-key', required=True, help='Путь к публичному ключу'
    )
    custom_enc_parser.add_argument(
        '--key-size', type=int, default=256,
        choices=[128, 192, 256],
        help='Размер симметричного ключа'
    )

    custom_dec_parser = subparsers.add_parser(
        'custom-dec', help='Дешифрование с пользовательскими ключами'
    )
    custom_dec_parser.add_argument(
        '--input', required=True, help='Путь к зашифрованному файлу'
    )
    custom_dec_parser.add_argument(
        '--output', required=True, help='Путь для сохранения дешифрованного файла'
    )
    custom_dec_parser.add_argument(
        '--priv-key', required=True, help='Путь к приватному ключу'
    )

    init_parser = subparsers.add_parser(
        'init', help='Создание файла настроек по умолчанию'
    )

    args = parser.parse_args()

    if args.command == 'init':
        create_default_config()
        return

    config = Config.from_json(args.settings)
    crypto_system = HybridCryptoSystem(config)

    match args.command:
        case 'gen':
            crypto_system.generate_and_save_keys(args.key_size)
        case 'enc':
            crypto_system.encrypt_file(args.input, args.output)
        case 'dec':
            crypto_system.decrypt_file(args.input, args.output)
        case 'custom-enc':
            crypto_system.encrypt_with_custom_keys(
                args.input, args.output, args.pub_key, args.key_size
            )
        case 'custom-dec':
            crypto_system.decrypt_with_custom_keys(
                args.input, args.output, args.priv_key
            )
        case _:
            print("Неизвестная команда")


if __name__ == "__main__":
    main()