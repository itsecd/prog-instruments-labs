import json
from assymetric import RSA
from symmetric import CAST_5
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key
from cryptography.hazmat.primitives import serialization


def read_json(path: str) -> dict:
    """
        Читает данные из JSON-файла и возвращает содержимое в виде словаря.
        path (str)- Путь к JSON-файлу для чтения.
        dict - Содержимое JSON-файла в виде словаря.
    """
    try:
        with open(path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
    except Exception as e:
        print(f"При чтении файла произошла ошибка: {str(e)}")


def write_file(path: str, data: str) -> None:
    """
       Записывает данные в файл.
       path (str) - Путь к файлу, в который нужно записать данные.
       data (str) - Строка данных для записи в файл.
    """
    try:
        with open(path, "a+", encoding='UTF-8') as file:
            file.write(data)
    except FileNotFoundError:
        print(f"Создан файл с названием: {path}")
    except Exception as e:
        print(f"Произошла ошибка при работе с файлом {path}: {e}")


def read_file(pathname: str) -> str:
    """
       Читает данные из файла и возвращает их в виде строки.
       pathname (str) - Путь к файлу для чтения.
       str - Содержимое файла в виде строки.
    """
    s = ''
    try:
        with open(pathname, 'r', encoding='utf-8') as file_read:
            s = file_read.read()
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
    return s


def write_bytes_to_file(path, bytes_) -> None:
    """
       Записывает байтовые данные в файл.
       path (str) - Путь к файлу, в который нужно записать байтовые данные.
       bytes_ (bytes) - Байтовые данные для записи в файл.
       """
    try:
        with open(path, "wb") as file:
            file.write(bytes_)
    except FileNotFoundError:
        print(f"Создан файл с названием: {path}")
    except Exception as e:
        print(f"Произошла ошибка при работе с файлом {path}: {e}")


def read_bytes_from_file(pathname) -> bytes:
    """
        Читает байтовые данные из файла и возвращает их.
        pathname (str): Путь к файлу для чтения.
        bytes: Байтовые данные из файла.
        """
    s = ''
    try:
        with open(pathname, 'rb') as file_read:
            s = file_read.read()
    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
    return s


def get_key_to_file(cast5: CAST_5, path: str) -> None:
    """
        Сохраняет ключ в файл.
        path (str) - Путь к файлу для сохранения ключа.
    """
    try:
        with open(path, 'wb') as key_file:
            key_file.write(cast5.key)
    except Exception as e:
        print(f"Произошла ошибка при сохранении ключа в файл: {e}")


def get_key_from_file(cast5: CAST_5, path: str) -> None:
    """
        Загружает ключ из файла.
        path (str): Путь к файлу с ключом.
    """
    try:
        with open(path, 'rb') as key_file:
            cast5.key = key_file.read()
    except FileNotFoundError:
        print(f"Файл с ключом не найден: {path}")
    except Exception as e:
        print(f"Произошла ошибка при загрузке ключа из файла: {e}")


def get_open_key_from_file(rsa: RSA, path_public) -> None:
    """
        Загружает открытый ключ RSA из файла.
        path_public (str) - Путь к файлу с открытым ключом.
    """
    try:
        with open(path_public, 'rb') as pem_in:
            public_bytes = pem_in.read()
        rsa.public_key = load_pem_public_key(public_bytes)
    except FileNotFoundError:
        print(f"Файл с открытым ключом не найден: {path_public}")
    except Exception as e:
        print(f"Произошла ошибка при загрузке открытого ключа: {e}")


def get_private_key_from_file(rsa: RSA, path_private) -> None:
    """
        Загружает закрытый ключ RSA из файла.
        path_private (str) - Путь к файлу с закрытым ключом.
    """
    try:
        with open(path_private, 'rb') as pem_in:
            private_bytes = pem_in.read()
        rsa.private_key = load_pem_private_key(private_bytes, password=None)
    except FileNotFoundError:
        print(f"Файл с закрытым ключом не найден: {path_private}")
    except Exception as e:
        print(f"Произошла ошибка при загрузке закрытого ключа: {e}")


def get_pub_and_priv_key_to_file(rsa: RSA, path_public, path_private) -> None:
    """
        Сохраняет открытый и закрытый ключи RSA в файлы.
        path_public (str) - Путь к файлу для сохранения открытого ключа.
        path_private (str) -  Путь к файлу для сохранения закрытого ключа.
    """
    try:
        if rsa.public_key is None:
            print('Ключи ещё не сгенерированы.')
            return
        with open(path_public, 'wb') as public_out:
            public_out.write(rsa.public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                         format=serialization.PublicFormat.SubjectPublicKeyInfo))
        with open(path_private, 'wb') as private_out:
            private_out.write(rsa.private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                            format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                            encryption_algorithm=serialization.NoEncryption()))
    except Exception as e:
        print(f'Произошла ошибка при сохранении ключей в файл: {e}')