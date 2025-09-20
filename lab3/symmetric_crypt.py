import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def read_file(file_path):
    """
    Читает содержимое файла в бинарном режиме с обработкой ошибок.
    """
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    except PermissionError:
        raise PermissionError(f"Нет доступа к файлу {file_path}.")
    except Exception as e:
        raise IOError(f"Ошибка при чтении файла {file_path}: {e}")


def write_file(file_path, data):
    """
    Записывает данные в файл в бинарном режиме с обработкой ошибок.
    """
    try:
        with open(file_path, 'wb') as f:
            f.write(data)
    except PermissionError:
        raise PermissionError(f"Нет доступа к файлу {file_path}.")
    except Exception as e:
        raise IOError(f"Ошибка при записи файла {file_path}: {e}")


def generate_symmetric_key():
    """
    Генерирует 128-битный ключ для SEED.
    """
    print("Генерация симметричного ключа SEED...")
    return os.urandom(16)


def encrypt_file(input_path, output_path, sym_key):
    """
    Шифрует файл с помощью SEED.
    """
    print(f"Шифрование файла {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Входной файл {input_path} не найден")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iv = os.urandom(16)
    cipher = Cipher(algorithms.SEED(sym_key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    plaintext = read_file(input_path)

    padder = padding.ANSIX923(128).padder()
    padded_text = padder.update(plaintext) + padder.finalize()

    ciphertext = encryptor.update(padded_text) + encryptor.finalize()

    try:
        write_file(output_path, iv + ciphertext)
        print(f"Зашифрованный файл сохранен в {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        raise


def decrypt_file(input_path, output_path, sym_key):
    """
    Расшифровывает файл с помощью SEED.
    """
    print(f"Расшифровка файла {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Зашифрованный файл {input_path} не найден")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = read_file(input_path)
    iv, ciphertext = data[:16], data[16:]

    cipher = Cipher(algorithms.SEED(sym_key), modes.CBC(iv))
    decryptor = cipher.decryptor()

    padded_text = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = padding.ANSIX923(128).unpadder()
    plaintext = unpadder.update(padded_text) + unpadder.finalize()

    try:
        write_file(output_path, plaintext)
        print(f"Расшифрованный файл сохранен в {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        raise

