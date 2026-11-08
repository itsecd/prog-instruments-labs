from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asymmetric_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from file_utils import read_binary_file, write_binary_file


def generate_rsa_key() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """
    Генерирует пару RSA-ключей.

    :return: закрытый и открытый RSA-ключи
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    return private_key, public_key


def save_private_key(private_key: rsa.RSAPrivateKey, private_key_path: str) -> None:
    """
    Сохраняет закрытый RSA-ключ.

    :param private_key: закрытый RSA-ключ
    :param private_key_path: путь для сохранения закрытого ключа
    :return: None
    """
    key_data = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    write_binary_file(private_key_path, key_data)


def save_public_key(public_key: rsa.RSAPublicKey, public_key_path: str) -> None:
    """
    Сохраняет открытый RSA-ключ.

    :param public_key: открытый RSA-ключ
    :param public_key_path: путь для сохранения открытого ключа
    :return: None
    """
    key_data = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    write_binary_file(public_key_path, key_data)


def load_private_key(private_key_path: str) -> rsa.RSAPrivateKey:
    """
    Загружает закрытый RSA-ключ.

    :param private_key_path: путь к закрытому RSA-ключу
    :return: закрытый RSA-ключ
    """
    try:
        key_data = read_binary_file(private_key_path)
        return load_pem_private_key(key_data, password=None)
    except ValueError as error:
        raise ValueError("Не удалось прочитать закрытый RSA-ключ.") from error


def encrypt_symmetric_key(key: bytes, public_key: rsa.RSAPublicKey) -> bytes:
    """
    Шифрует симметричный ключ открытым RSA-ключом.

    :param key: симметричный ключ
    :param public_key: открытый RSA-ключ
    :return: зашифрованный симметричный ключ
    """
    return public_key.encrypt(
        key,
        asymmetric_padding.OAEP(
            mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def decrypt_symmetric_key(encrypted_key: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
    """
    Расшифровывает симметричный ключ закрытым RSA-ключом.

    :param encrypted_key: зашифрованный симметричный ключ
    :param private_key: закрытый RSA-ключ
    :return: симметричный ключ
    """
    return private_key.decrypt(
        encrypted_key,
        asymmetric_padding.OAEP(
            mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
