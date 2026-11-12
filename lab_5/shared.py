from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.padding import OAEP, MGF1
import os


def serialize(path: str, data: bytes) -> None:
    """
    Функция для сериализации данных в файл
    Args:
        path (str): Путь, по которому будут сохранены данные
    """
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, mode='wb') as file:
        file.write(data)


def deserialize(path: str) -> bytes:
    """
    Функция для считывания (десериализации) данных из файла
    Args:
        path (str): Путь до считываемого файла
    Returns:
        Байты - данные из считанного фалы
    """
    with open(path, mode='rb') as file:
        data = file.read()
    return data


def get_symmetric_key(symmetric_key_path: str, secret_key_path: str) -> bytes:
    """
    Возвращает расшифрованный ключ для симметричного алгоритма системы
    Args:
        symmetric_key_path (str): Путь до зашифрованного симметричного ключа
        secret_key_path (str): Путь до приватного ключа RSA
    Returns:
        Расшифрованный симметричный ключ в виде набора байт
    """
    encrypted_key = deserialize(symmetric_key_path)
    private_key = load_pem_private_key(deserialize(secret_key_path), password=None)

    symmetric_key = private_key.decrypt(encrypted_key,
        padding=OAEP(mgf=MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )
    return symmetric_key