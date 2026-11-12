from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.padding import OAEP, MGF1
from colorama import Fore, Back, Style, init
import os
from shared import serialize


def generate_symmetric_key(size: int) -> bytes:
    """
    Функция, генерирующая ключ для симметричного алгоритма шифрования.
    Args:
        size (int): Размер ключа в битах
    Returns:
        Ключ в виде набора байт
    """
    return os.urandom(size // 8)


def generate_keys(settings: dict[str, str | int | bytes]) -> None:
    """
    Процедура, создающая ключи и сериализующая их
    Args:
        settings (dict): Параметры приложения
    """
    init(autoreset=True)
    print(Back.GREEN + Style.BRIGHT + f"Создаём симметричный ключ длинной {settings['3des_key_size']} бит.")
    symmetric_key = generate_symmetric_key(settings["3des_key_size"])

    print(Back.GREEN + Style.BRIGHT + f"Создаём публичный и приватный ключ.")
    keys = rsa.generate_private_key(public_exponent=settings["rsa_public_exponent"], key_size=settings["rsa_key_size"])
    private_key = keys
    public_key = keys.public_key()

    serialize(settings["secret_key"],
              private_key.private_bytes(
                  encoding=Encoding.PEM,
                  format=PrivateFormat.TraditionalOpenSSL,
                  encryption_algorithm=NoEncryption())
              )
    print("Приватный ключ сохранён в", Fore.BLUE + f"{os.path.abspath(settings['secret_key'])}")

    serialize(settings["public_key"],
              public_key.public_bytes(encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo)
              )
    print("Публичный ключ сохранён в", Fore.BLUE + f"{os.path.abspath(settings['public_key'])}")
    print(Back.GREEN + Style.BRIGHT + "Шифруем симметричный ключ.")
    encrypted_key = public_key.encrypt(symmetric_key,
        padding=OAEP(mgf=MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
    )

    serialize(settings["symmetric_key"], encrypted_key)
    print("Зашифрованный симметричный ключ сохранён в", Fore.BLUE + f"{os.path.abspath(settings['symmetric_key'])}")