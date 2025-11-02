import argparse
from enum import Enum, auto
from loguru import logger

from hybrid_system import decrypt_data, encrypt_data, generate_keys
from utils import load_settings

# Определение Enum для режимов работы
class Mode(Enum):
    GENERATION = auto()
    ENCRYPTION = auto()
    DECRYPTION = auto()

def main():
    """
    Processes command line arguments and performs the selected operation.
    """
    parser = argparse.ArgumentParser(description='Hybrid cryptosystem')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='settings.json',
        help='Path to JSON settings file (default: settings.json)'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-gen',
        '--generation',
        action='store_true',
        help='Starts key generation mode'
    )
    group.add_argument(
        '-enc',
        '--encryption',
        action='store_true',
        help='Starts encryption mode'
    )
    group.add_argument(
        '-dec',
        '--decryption',
        action='store_true',
        help='Starts decryption mode'
    )

    args = parser.parse_args()

    try:
        logger.info("Starting hybrid cryptosystem")
        logger.debug(f"Command line arguments: {args}")

        # Загрузка настроек из файла
        settings = load_settings(args.config)
        logger.debug("Settings loaded successfully")

        # Выбор действия с помощью match/case на основе аргументов
        match True:
            case args.generation:
                logger.info("Starting key generation mode")
                generate_keys(settings)
            case args.encryption:
                logger.info("Starting encryption mode")
                encrypt_data(settings)
            case args.decryption:
                logger.info("Starting decryption mode")
                decrypt_data(settings)
            case _:
                logger.error("Operating mode not selected")
                raise ValueError("Operating mode not selected.")

        logger.success("Operation completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        exit(1)
    except ValueError as e:
        logger.error(f"Value error: {e}")
        exit(1)
    except KeyError as e:
        logger.error(f"Key error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        exit(1)

if __name__ == '__main__':
    main()