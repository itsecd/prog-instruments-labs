import argparse
import sys
import logging
from sym import *
from asym import *
from file_utils import read, save
from crypt_utils import read_binary_file, write_binary_file, load_json_config
from enum import Enum, auto

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ProgramMode(Enum):
    GENERATE = auto()
    ENCRYPT = auto()
    DECRYPT = auto()
    ENCRYPT_KEY = auto()


def parse_args():
    """Парсит аргументы командной строки и определяет режим работы"""
    logger.debug("Парсинг аргументов командной строки")
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-gen', '--generate', action='store_true',
                       help='Генерация новых ключей')
    group.add_argument('-enc', '--encrypt', action='store_true',
                       help='Шифрование файла')
    group.add_argument('-dec', '--decrypt', action='store_true',
                       help='Дешифрование файла')
    group.add_argument('-enc-key', '--encrypt-key', metavar='KEY_FILE',
                       help='Зашифровать существующий симметричный ключ')

    parser.add_argument('--sym-key', metavar='PATH',
                        help='Путь к симметричному ключу (по умолчанию из settings.json)')

    args = parser.parse_args()

    match args:
        case _ if args.generate:
            logger.info("Выбран режим: GENERATE")
            return ProgramMode.GENERATE, args
        case _ if args.encrypt:
            logger.info("Выбран режим: ENCRYPT")
            return ProgramMode.ENCRYPT, args
        case _ if args.decrypt:
            logger.info("Выбран режим: DECRYPT")
            return ProgramMode.DECRYPT, args
        case _ if args.encrypt_key:
            logger.info(f"Выбран режим: ENCRYPT_KEY, файл ключа: {args.encrypt_key}")
            return ProgramMode.ENCRYPT_KEY, args
        case _:
            logger.warning("Неизвестный режим работы")
            parser.print_help()
            sys.exit(1)


def setup_keys(enc_key_path: str, key_size: int,
               public_key_path: str, private_key_path: str):
    """Генерирует и сохраняет криптографические ключи"""
    try:
        logger.info(f"Генерация ключей: sym_key={enc_key_path}, size={key_size}")
        cast_key = create_symmetric_components(key_size)
        priv_key, pub_key = generate_rsa_keys()
        logger.debug("Ключи успешно сгенерированы")

        save_key_to_pem(pub_key, public_key_path, False)
        save_key_to_pem(priv_key, private_key_path, True)
        logger.debug("Ключи сохранены в PEM формате")

        encrypted_key = rsa_encrypt(pub_key, cast_key)
        write_binary_file(enc_key_path, encrypted_key)
        logger.debug("Симметричный ключ зашифрован и сохранен")

        logger.info("Ключи успешно созданы и сохранены")
        print("Ключи успешно созданы и сохранены")
    except Exception as e:
        logger.error(f"Ошибка при генерации ключей: {e}")
        print(f"Ошибка при генерации ключей: {e}")
        sys.exit(1)


def encrypt_data(input_path: str, priv_key_path: str,
                 enc_key_path: str, output_path: str):
    """Шифрует данные гибридной системой"""
    try:
        logger.info(f"Шифрование данных: input={input_path}, output={output_path}")
        text_data = read(input_path)
        logger.debug(f"Исходные данные прочитаны, размер: {len(text_data)} символов")

        priv_key = load_key_from_pem(priv_key_path, True)
        enc_key = read_binary_file(enc_key_path)
        logger.debug("Ключи успешно загружены")

        sym_key = rsa_decrypt(priv_key, enc_key)
        encrypted_data = encrypt_with_cast5(sym_key, text_data)
        logger.debug("Данные успешно зашифрованы")

        write_binary_file(output_path, encrypted_data)
        logger.info(f"Данные успешно зашифрованы и сохранены в {output_path}")
        print("Данные успешно зашифрованы")
    except Exception as e:
        logger.error(f"Ошибка при шифровании: {e}")
        print(f"Ошибка при шифровании: {e}")
        sys.exit(1)


def decrypt_data(input_path: str, priv_key_path: str,
                 enc_key_path: str, output_path: str):
    """Расшифровывает данные гибридной системой"""
    try:
        logger.info(f"Дешифрование данных: input={input_path}, output={output_path}")
        priv_key = load_key_from_pem(priv_key_path, True)
        enc_key = read_binary_file(enc_key_path)
        enc_data = read_binary_file(input_path)
        logger.debug("Все необходимые данные загружены")

        sym_key = rsa_decrypt(priv_key, enc_key)
        result = decrypt_with_cast5(sym_key, enc_data)
        logger.debug("Данные успешно расшифрованы")

        save(output_path, result)
        logger.info(f"Данные успешно расшифрованы и сохранены в {output_path}")
        print("Данные успешно расшифрованы")
    except Exception as e:
        logger.error(f"Ошибка при дешифровании: {str(e)}")
        print(f"Ошибка при дешифровании: {str(e)}")
        sys.exit(1)


def encrypt_existing_key(key_path: str, pub_key_path: str, output_path: str) -> None:
    """Шифрует существующий симметричный ключ с помощью RSA"""
    try:
        logger.info(f"Шифрование ключа: {key_path} -> {output_path}")
        key_data = read_binary_file(key_path)
        pub_key = load_key_from_pem(pub_key_path, False)
        logger.debug("Ключи успешно загружены")

        encrypted_key = rsa_encrypt(pub_key, key_data)
        write_binary_file(output_path, encrypted_key)

        logger.info(f"Ключ успешно зашифрован и сохранен в {output_path}")
        print(f"Ключ успешно зашифрован и сохранен в {output_path}")
    except Exception as e:
        logger.error(f"Ошибка при шифровании ключа: {e}")
        print(f"Ошибка при шифровании ключа: {e}")
        sys.exit(1)


def main():
    """Запускает выбранный режим"""
    try:
        logger.info("Запуск программы")
        mode, args = parse_args()
        config = load_json_config('settings.json')['settings']
        sym_key_path = args.sym_key if args.sym_key else config['sym_key']
        logger.debug(f"Конфигурация загружена, sym_key_path: {sym_key_path}")

        match mode:
            case ProgramMode.GENERATE:
                setup_keys(
                    config['sym_key'],
                    int(config['len_key']),
                    config['publ_key'],
                    config['priv_key']
                )
            case ProgramMode.ENCRYPT:
                encrypt_data(
                    config['orig_file'],
                    config['priv_key'],
                    sym_key_path,
                    config['enc_file']
                )
            case ProgramMode.DECRYPT:
                decrypt_data(
                    config['enc_file'],
                    config['priv_key'],
                    sym_key_path,
                    config['dec_file']
                )
            case ProgramMode.ENCRYPT_KEY:
                encrypt_existing_key(
                    args.encrypt_key,
                    config['publ_key'],
                    sym_key_path
                )
            case _:
                logger.error("Неизвестный режим работы")
                print("Неизвестный режим работы")
                sys.exit(1)
        logger.info("Программа завершена успешно")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()