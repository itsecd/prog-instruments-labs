import json
import logging

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key


def deserialize_private(path_to_private_key: str) -> rsa.RSAPrivateKey:
    """
    Функция десиарилизует приватный ключ из указанного файла(path_to_private_key).
    @param path_to_private_key: путь до файла с сохранненым приватным ключом. Тип str.
    @return d_private_key: приватный ключ. Тип rsa.RSAPrivateKey.
    """
    try:
        with open(path_to_private_key, 'rb') as pem_in:
            private_bytes = pem_in.read()
        d_private_key = load_pem_private_key(private_bytes, password=None, )
        logging.info("Функция deserialize_private в serialization_deserialitation_and_text.py десиарилизовала приватный ключ.")
        return d_private_key
    except FileNotFoundError:
        print("Файл не найден.")
        raise
    except Exception as e:
        print(f"Произошла ошибка deserialize_private: {e}")
        raise


def deserialize_public(path_to_public_key: str) -> rsa.RSAPublicKey:
    """
    Функция десиарилизует публичный ключ из указанного файла(path_to_public_key).
    @param path_to_public_key: путь до файла с сохранненым публичным ключом. Тип str.
    @return d_public_key: публичный ключ. Тип rsa.RSAPublicKey.
    """
    try:
        with open(path_to_public_key, 'rb') as pem_in:
            public_bytes = pem_in.read()
        d_public_key = load_pem_public_key(public_bytes)
        logging.info(
            "Функция deserialize_public в serialization_deserialitation_and_text.py десиарилизовала публичный ключ.")
        return d_public_key
    except FileNotFoundError:
        print("Файл не найден.")
        raise
    except Exception as e:
        print(f"Произошла ошибка deserialize_public: {e}")
        raise


def serialize_private(path_to_private_key: str, private_key: rsa.RSAPrivateKey) -> None:
    """
    Функция сериализует приватный ключ(private_key) по заданному пути(path_to_private_key).
    @param path_to_private_key: путь до файла для сохранения приватного ключа. Тип str.
    @param private_key: приватный ключ, который нужно сериализовать. Тип rsa.RSAPrivateKey.
    """
    try:
        with open(path_to_private_key, 'wb') as private_out:
            private_out.write(private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                        encryption_algorithm=serialization.NoEncryption()))
        logging.info(
            "Функция serialize_private в serialization_deserialitation_and_text.py сиарилизовала приватный ключ.")
    except FileNotFoundError:
        print("Файл не найден.")
        raise
    except Exception as e:
        print(f"Произошла ошибка serialize_private: {e}")
        raise


def serialize_public(path_to_public_key: str, public_key: rsa.RSAPublicKey) -> None:
    """
    Функция сериализует публичный ключ(public_key) по заданному пути(path_to_public_key).
    @param path_to_public_key: путь до файла для сохранения публичного ключа. Тип str.
    @param public_key: публичный ключ, который нужно сериализовать. Тип rsa.RSAPublicKey.
    """
    try:
        with open(path_to_public_key, 'wb') as public_out:
            public_out.write(public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                     format=serialization.PublicFormat.SubjectPublicKeyInfo))
            logging.info("Функция serialize_public в serialization_deserialitation_and_text.py сиарилизовала публичный ключ.")
    except FileNotFoundError:
        print("Файл не найден.")
        raise
    except Exception as e:
        print(f"Произошла ошибка serialize_public: {e}")
        raise


def serialize_asymmetric_keys(path_to_private_key: str, path_to_public_key: str, private_key: rsa.RSAPrivateKey,
                              public_key: rsa.RSAPublicKey) -> None:
    """
    Функция сериализует приватный ключ(private_key) и публичный ключ(public_key) по заданным путям
    (path_to_private_key) и (path_to_public_key).
    @param path_to_private_key: путь до файла для сохранения приватного ключа. Тип str.
    @param path_to_public_key: приватный ключ, который нужно сериализовать. Тип str.
    @param private_key: приватный ключ, который нужно сериализовать. Тип rsa.RSAPrivateKey.
    @param public_key: публичный ключ, который нужно сериализовать. Тип rsa.RSAPublicKey.
    """
    serialize_private(path_to_private_key, private_key)
    serialize_public(path_to_public_key, public_key)
    logging.info("Функция serialize_asymmetric_keys в serialization_deserialitation_and_text.py сиарилизовала приватный и публичный ключ.")


def serialize_symmetric_key(path_to_symmetric_key: str, symmetric_key: bytes) -> None:
    """
    Функция сериализует симметричный ключ(symmetric_key) в файл path_to_symmetric_key.
    @param path_to_symmetric_key: путь до файла для симмитричного ключа. Тип str.
    @param symmetric_key: симметричный ключ. Тип bytes.
    """
    try:
        with open(path_to_symmetric_key, 'wb') as file:
            file.write(symmetric_key)
        logging.info("Функция serialize_symmetric_key в serialization_deserialitation_and_text.py сиарилизовала симметричный ключ.")
    except FileNotFoundError:
        logging.error("Файл не найден")
        raise
    except Exception as e:
        logging.error(f"Произошла ошибка в serialize_symmetric_key: {e}")
        raise


def deserialize_symmetric_key(path_to_symmetric_key: str) -> bytes:
    """
    Функция десериализует симметричный ключ из файла path_to_symmetric_key.
    @param path_to_symmetric_key: путь до файла с симметричным ключом. Тип str.
    @return symmetric_key: симметричный ключ. Тип bytes.
    """
    try:
        with open(path_to_symmetric_key, mode='rb') as key_file:
            symmetric_key = key_file.read()
        logging.info("Функция deserialize_symmetric_key в serialization_deserialitation_and_text.py десиарилизовала симметричный ключ.")
        return symmetric_key
    except FileNotFoundError:
        logging.error("Файл не найден")
        raise
    except Exception as e:
        logging.error(f"Произошла ошибка в deserialize_symmetric_key: {e}")
        raise


def save_text(file_name: str, text: bytes):
    """
    Функция записывает текст (text) в файл с названием file_name.
    @param file_name: название файла для записи текста. Тип str.
    @param text: текст для сохранения  в байтах. Тип bytes.
    """
    try:
        with open(file_name, 'wb') as file:
            file.write(text)
        logging.info("Функция save_text в serialization_deserialitation_and_text.py сохранила текст.")

    except FileNotFoundError:
        logging.error("Файл не найден")
        raise
    except Exception as e:
        logging.error(f"Произошла ошибка в save_text: {e}")
        raise


def save_text_str(file_name: str, text: str):
    """
    Функция записывает текст (text) в файл с названием file_name.
    @param file_name: название файла для записи текста. Тип str.
    @param text: текст для сохранения  в.виде строки Тип str.
    """
    try:
        with open(file_name, 'w') as file:
            file.write(text)
        logging.info("Функция save_text_str в serialization_deserialitation_and_text.py сохранила текст.")
    except FileNotFoundError:
        logging.error("Файл не найден")
        raise
    except Exception as e:
        logging.error(f"Произошла ошибка в save_text_str: {e}")
        raise


def read_text(file_name: str):
    """
    Функция считывает текст из файла с названием file_name. Затем возвращает считанный текст.
    @param file_name: название файла для считывания.Тип str.
    @return content: содержимое файла. Тип str.
    """
    try:
        with open(file_name, mode='rb') as key_file:
            content = key_file.read()
        logging.info("Функция read_text в serialization_deserialitation_and_text.py считала текст.")
        return content
    except FileNotFoundError:
        logging.error("Файл не найден")
        raise
    except Exception as e:
        logging.error(f"Произошла ошибка в read_text: {e}")
        raise


def read_json_file(file_path: str) -> dict:
    """
    Функция считывает данные из JSON файла.
    :param file_path: указывает на расположение JSON файла.
    :return dict:
    """
    try:
        with open(file_path, "r", encoding="UTF-8") as file:
            json_data = json.load(file)
            logging.info("Функция read_json_file в serialization_deserialitation_and_text.py считала данные из JSON файла.")
            return json_data
    except FileNotFoundError:
        logging.error("Файл не найден")
        raise
    except json.JSONDecodeError:
        logging.error("Ошибка при считывании JSON-данных.")
        raise
    except Exception as e:
        logging.error(f"Произошла ошибка в read_json_file: {e}")
        raise
