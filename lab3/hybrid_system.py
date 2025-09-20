import json
from lab3.asymmetric_crypt import decrypt_symmetric_key, encrypt_symmetric_key, generate_asymmetric_keys
from lab3.symmetric_crypt import decrypt_file, encrypt_file, generate_symmetric_key
from lab3.utils import serialize_key


def load_settings(config_file: str) -> dict:
    """
    Загружает настройки из JSON-файла.
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл настроек '{config_file}' не найден.")
        raise
    except json.JSONDecodeError:
        print(f"Ошибка: Неверный формат JSON в файле '{config_file}'.")
        raise
    except Exception as e:
        print(f"Произошла ошибка при загрузке настроек: {e}")
        raise


def validate_settings(settings: dict, required_keys: list) -> None:
    """
    Проверяет наличие необходимых ключей в настройках.
    """
    missing_keys = [key for key in required_keys if key not in settings]
    if missing_keys:
        raise KeyError(f"Отсутствуют необходимые ключи: {', '.join(missing_keys)}")


def generate_keys(settings: dict) -> None:
    """
    Генерирует и сохраняет ключи, указанные в настройках.
    """
    try:
        required_keys = settings.get('required_keys', {}).get('generate', [])
        validate_settings(settings, required_keys)

        sym_key = generate_symmetric_key()
        private_key, public_key = generate_asymmetric_keys()

        serialize_key(public_key, settings[required_keys[0]], 'public')  # public_key
        serialize_key(private_key, settings[required_keys[1]], 'private')  # secret_key
        encrypt_symmetric_key(sym_key, public_key, settings[required_keys[2]])  # symmetric_key
        print('Генерация ключей завершена.')

    except KeyError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка при генерации ключей: {e}")


def encrypt_data(settings: dict) -> None:
    """
    Шифрует данные.
    """
    try:
        required_keys = settings.get('required_keys', {}).get('encrypt', [])
        validate_settings(settings, required_keys)

        sym_key = decrypt_symmetric_key(
            settings[required_keys[0]],  
            settings[required_keys[1]]   
        )
        encrypt_file(
            settings[required_keys[2]],
            settings[required_keys[3]],
            sym_key
        )
        print('Шифрование данных завершено.')

    except KeyError as e:
        print(f"Ошибка: {e}")
    except FileNotFoundError:
        print("Ошибка: Один из указанных файлов не найден.")
    except Exception as e:
        print(f"Произошла ошибка при шифровании данных: {e}")


def decrypt_data(settings: dict) -> None:
    """
    Расшифровывает данные.
    """
    try:
        required_keys = settings.get('required_keys', {}).get('decrypt', [])
        validate_settings(settings, required_keys)

        sym_key = decrypt_symmetric_key(
            settings[required_keys[0]], 
            settings[required_keys[1]]  
        )
        decrypt_file(
            settings[required_keys[2]],  
            settings[required_keys[3]], 
            sym_key
        )
        print('Расшифровка данных завершена.')

    except KeyError as e:
        print(f"Ошибка: {e}")
    except FileNotFoundError:
        print("Ошибка: Один из указанных файлов не найден.")
    except Exception as e:
        print(f"Произошла ошибка при расшифровке данных: {e}")
