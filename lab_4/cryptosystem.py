import logging

from asymmetric_key import AsymmetricKey
from symmetric_key import SymmetricKey
from serialization_deserialitation_and_text import read_text, save_text, save_text_str, deserialize_symmetric_key, \
    serialize_symmetric_key, serialize_asymmetric_keys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

class Cryptosystem:
    """
    Class for Cryptosystem:
    @methods:
        __init__:
        generate_keys:
        encrypt:
        decrypt:
    """

    def __init__(self, number_of_bits: int):
        """
        Конструктор для класса.
        атрибуты:
        symmetric - объект класса SymmetricKey()
        asymmetric - объект класса AsymmetricKey()
        number_of_bits - число битов для шифрования.
        @param number_of_bits:
        """
        self.symmetric = SymmetricKey()
        self.asymmetric = AsymmetricKey()
        self.number_of_bits = number_of_bits
        logging.info("Криптосистема в cryptosystem.py успешно создана.")

    def generate_keys(self, path_to_symmetric_key: str, path_to_public_key: str, path_to_private_key: str) -> None:
        """
        Метод генерирует ключи для симметричнего и асимметричного алгоритма шифрования и
        сохраняет их по указанным путям.
        @param path_to_symmetric_key: путь до файла с симметричным ключом. Тип str.
        @param path_to_public_key: путь до файла с публичным ключом. Тип str.
        @param path_to_private_key: путь до файла с приватным ключом. Тип str.
        """
        symmetric_key = self.symmetric.generate_key(self.number_of_bits)
        keys = self.asymmetric.generate_keys()
        serialize_asymmetric_keys(path_to_private_key, path_to_public_key, keys[0], keys[1])
        symmetric_key_encrypted = self.asymmetric.encrypt_text(symmetric_key, keys[1])
        serialize_symmetric_key(path_to_symmetric_key, symmetric_key_encrypted)
        logging.info("Метод generate_keys в Сryptosystem успешно создал ключи")

    def encrypt(self, path_to_text_for_encryption: str, path_to_symmetric_key: str, path_to_private_key: str,
                path_to_save_encrypted_text: str) -> None:
        """
        Метод шифрует переданный текст из файла path_to_text_for_encryption с помощью
        симметричного и приватного ключей из файлов path_to_symmetric_key и path_to_private_key,
        после сохраняет зашифрованный текст в файле path_to_save_encrypted_text.
        @param path_to_text_for_encryption: путь до текста, который нужно зашифровать. Тип str.
        @param path_to_symmetric_key: Путь до файла с симметричным ключом. Тип str.
        @param path_to_private_key: Путь до файла с приватным ключом. Тип str.
        @param path_to_save_encrypted_text: Путь для сохранения зашифрованного сообщения. Тип str.
        """
        encrypted_symmetric_key = deserialize_symmetric_key(path_to_symmetric_key)
        decrypted_symmetric_key = self.asymmetric.decrypt_text(path_to_private_key, encrypted_symmetric_key)
        text = read_text(path_to_text_for_encryption)
        encrypted_text = self.symmetric.encrypt_symmetric(text, decrypted_symmetric_key, self.number_of_bits)
        save_text(path_to_save_encrypted_text, encrypted_text)
        logging.info("Метод encrypt в Сryptosystem успешно зашифровал текст")

    def decrypt(self, path_to_encrypted_text: str, path_to_symmetric_key: str, path_to_private_key: str,
                path_to_save_decrypted_text: str) -> None:
        """
        Метод дешифрует переданный текст из файла path_to_encrypted_text с помощью
        симметричного и приватного ключей из файлов path_to_symmetric_key и path_to_private_key,
        после чего сохраняет его в файле path_to_save_decrypted_text.
        @param path_to_encrypted_text: Путь до файла для дешифрования. Тип str.
        @param path_to_symmetric_key: Путь до файла с симметричным ключом. Тип str.
        @param path_to_private_key: Путь до файла с приватным ключом. Тип str.
        @param path_to_save_decrypted_text: Путь для сохранения дешифрованного текста. Тип str.
        """
        encrypted_symmetric_key = deserialize_symmetric_key(path_to_symmetric_key)
        decrypted_symmetric_key = self.asymmetric.decrypt_text(path_to_private_key, encrypted_symmetric_key)
        encrypted_text = read_text(path_to_encrypted_text)
        decrypted_text = self.symmetric.decrypt_symmetric(encrypted_text, decrypted_symmetric_key, self.number_of_bits)
        save_text_str(path_to_save_decrypted_text, decrypted_text)
        logging.info("Метод decrypt в Сryptosystem успешно расшифровал текст")
