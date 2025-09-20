import argparse
from enum import Enum, auto

from lab3.hybrid_system import decrypt_data, encrypt_data, generate_keys
from lab3.utils import load_settings

# Определение Enum для режимов работы
class Mode(Enum):
    GENERATION = auto()
    ENCRYPTION = auto()
    DECRYPTION = auto()

def main():
    """
    Обрабатывает аргументы командной строки и выполняет выбранную операцию.
    """
    parser = argparse.ArgumentParser(description='Гибридная криптосистема')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='settings.json',
        help='Путь к файлу настроек JSON (по умолчанию: settings.json)'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-gen',
        '--generation',
        action='store_true',
        help='Запускает режим генерации ключей'
    )
    group.add_argument(
        '-enc',
        '--encryption',
        action='store_true',
        help='Запускает режим шифрования'
    )
    group.add_argument(
        '-dec',
        '--decryption',
        action='store_true',
        help='Запускает режим расшифровки'
    )

    args = parser.parse_args()

    try:
        # Загрузка настроек из файла
        settings = load_settings(args.config)

        # Выбор действия с помощью match/case на основе аргументов
        match True:
            case args.generation:
                generate_keys(settings)
            case args.encryption:
                encrypt_data(settings)
            case args.decryption:
                decrypt_data(settings)
            case _:
                raise ValueError("Не выбран режим работы.")

    except FileNotFoundError as e:
        print(f'Ошибка: {e}')
        exit(1)
    except ValueError as e:
        print(f'Ошибка: {e}')
        exit(1)
    except KeyError as e:
        print(f'Ошибка: {e}')
        exit(1)
    except Exception as e:
        print(f'Произошла ошибка: {e}')
        exit(1)

if __name__ == '__main__':
    main()
