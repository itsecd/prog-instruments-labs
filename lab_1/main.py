import argparse
import os
import sys
from pathlib import Path
from hybrid import Hybrid
from asymmetrical import Asymmetrical
from filehandler import FileHandler
from constants import SETTINGS_FILE, DEFAULT_SETTINGS


def create_default_settings_if_needed() -> None:
    """
    Создает файл настроек с дефолтными значениями, если он отсутствует
    :return: None
    """
    if not Path(SETTINGS_FILE).exists():
        os.makedirs("keys", exist_ok=True)
        os.makedirs("texts", exist_ok=True)
        FileHandler.write_json(SETTINGS_FILE, DEFAULT_SETTINGS)
        print(f"Создан {SETTINGS_FILE} с настройками по умолчанию.")




def load_settings() -> dict[str, str]:
    """
    Загружает настройки из JSON файла
    :return: настройки из JSON файла
    """
    try:
        settings = FileHandler.get_json(SETTINGS_FILE)
        return settings
    except Exception as e:
        print(f"Ошибка загрузки настроек: {e}")
        print("Используются значения по умолчанию")
        return DEFAULT_SETTINGS






def parse_arguments() -> argparse.Namespace:
    """
    Парсинг аргументов командной строки
    :return: объект с атрибутами
    """
    parser = argparse.ArgumentParser(
        description="Гибридная криптосистема (RSA + CAST5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["generate", "encrypt", "decrypt"],
        required=True,#аргумент обязателен
        help="Выберите режим работы:\ngenerate - создание ключей"
             "\nencrypt - зашифровать текст"
             "\ndecrypt - расшифровать текст"
    )
    parser.add_argument(
        "-l", "--key-length",
        type=int,
        choices=[i for i in range(40, 129) if i % 8 == 0],
        metavar="[40-128]",
        default=128,
        help="Длина симметричного ключа в битах (только для режима generate)"
    )
    return parser.parse_args()


def main():
    """
    Основной управляющий модуль
    :return: None
    """
    create_default_settings_if_needed()#Создание дефолтных настроек
    settings = load_settings()
    args = parse_arguments()

    try:
        match args.mode:
            case "generate":
                print("Генерация ключей...")

                p, pub, s = Hybrid.generate_keys(args.key_length)

                encr = Asymmetrical.encrypt_by_public_key(pub, s)

                FileHandler.serialize_public_key(settings["public_key"], pub)
                FileHandler.serialize_private_key(settings["private_key"], p)
                FileHandler.serial_sym_key(settings["symmetric_key"], encr)

                print("Ключи успешно сгенерированы и сохранены")

            case "encrypt":
                print("Шифрование данных...")

                if not all([
                    os.path.exists(settings["public_key"]),
                    os.path.exists(settings["private_key"]),
                    os.path.exists(settings["symmetric_key"])
                ]):
                    raise FileNotFoundError(
                        "Ключи не найдены. Сначала выполните "
                        "генерацию ключей (режим generate)"
                    )

                if not os.path.exists(settings["initial_text"]):
                    raise FileNotFoundError(
                        f"Исходный файл {settings['initial_text']} не найден"
                    )

                pr_key = FileHandler.deserial_pr_key(settings["private_key"])
                enc_sym_key = FileHandler.deserial_sym_key(settings["sym_key"])
                plaintext = FileHandler.read_txt(settings["initial_text"])

                enc_data = Hybrid.encrypt_data(
                    private_key=pr_key,
                    encrypted_sym_key=enc_sym_key,
                    plaintext=plaintext
                )

                FileHandler.serial_sym_key(settings["enc_text"], enc_data)
                print(f"Данные зашифрованы в {settings['encrypted_text']}")

            case "decrypt":
                print("Дешифрование данных...")

                if not os.path.exists(settings["encrypted_text"]):
                    raise FileNotFoundError(
                        f"Зашифрованный файл "
                        f"{settings['encrypted_text']} не найден"
                    )

                if not all([
                    os.path.exists(settings["private_key"]),
                    os.path.exists(settings["symmetric_key"])
                ]):
                    raise FileNotFoundError(
                        "Необходимые ключи не найдены. "
                        "Проверьте наличие приватного ключа и "
                        "зашифрованного симметричного ключа"
                    )

                pr_key = FileHandler.des_pr_key(settings["private_key"])
                enc_sym_key = FileHandler.des_sym_key(settings["sym_key"])
                enc_data = FileHandler.des_sym_key(settings["encrypted_text"])

                decr_text = Hybrid.decrypt_data(
                    private_key=pr_key,
                    encrypted_sym_key=enc_sym_key,
                    encrypted_data=enc_data
                )

                FileHandler.write_txt(settings["decrypted_text"], decr_text)
                print(f"Данные дешифрованы в {settings['decrypted_text']}")

            case _:
                print("Неизвестная команда")
    except Exception as e:
        print(f"Ошибка: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()