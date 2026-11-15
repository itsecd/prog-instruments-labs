import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key

def load_json(path)-> dict:
    """Загрузка настроек программы"""
    try:
        with open(path) as json_file:
            json_data = json.load(json_file)
            return json_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")

def save_asy_key(public_path,private_path,private_key,public_key)->None:
    """Сохранение открытых и закрытых ключей """
    try:
        with open(public_path, 'wb') as public_out:
            public_out.write(public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                     format=serialization.PublicFormat.SubjectPublicKeyInfo))
        with open(private_path, 'wb') as private_out:
            private_out.write(private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                        encryption_algorithm=serialization.NoEncryption()))
    except IOError as e:
        print(f"Ошибка сохранения ключей RSA {e}")

def save_sym_key(path,key)->None:
    """Сохранение ключа симметричного алгоритма"""
    try:
        with open(path, 'wb') as key_file:
            key_file.write(key)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")

def load_sym_key(path)->bytes:
    """Загрузка ключа симметричного алгоритма"""
    try:
        with open(path, mode='rb') as key_file: 
            return key_file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")

def load_asy_pub_key(path)->RSAPublicKey:
    """Загрузка открытого ключа"""
    try:
        with open(path, 'rb') as pem_in:
            public_bytes = pem_in.read()
        d_public_key = load_pem_public_key(public_bytes)
        return d_public_key
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")

def load_asy_pri_key(path)->RSAPrivateKey:
    """Загрузка закрытого ключа"""
    try:
        with open(path, 'rb') as pem_in:
            private_bytes = pem_in.read()
        d_private_key = load_pem_private_key(private_bytes,password=None)
        return d_private_key
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")


def write_text_file(text, path)->None:
    """Запись текста в файл"""
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(text)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ошибка записи файла по пути {path}")    

def read_text_file(path)->str:
    """Чтение текста из файла"""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")

def write_file(text, path)->None:
    """Запись бинарных данных в файл"""
    try:
        with open(path, 'wb') as file:
            file.write(text)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ошибка записи файла по пути {path}")

def read_file(path)->bytes:
    """Чтение бинарных данных из файл"""
    try:
        with open(path, 'rb') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {path} не найден")

    