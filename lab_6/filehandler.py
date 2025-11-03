import json
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key

class FileHandler:
    """
    Вспомогательный класс для работы с файлами
    """
    @staticmethod
    def serialize_public_key(file_path: str, public_key: RSAPublicKey) -> None:
        """
        Сериализация открытого ключа в pem файл
        :param file_path: путь для сохранения
        :param public_key: открытый ключ
        :return: None
        """
        try:
            with open(file_path, 'wb') as public_out:
                public_out.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        except Exception as e:
            raise Exception(f"Ошибка записи публичного ключа: {str(e)}")

    @staticmethod
    def serialize_private_key(file_path: str, private_key: RSAPrivateKey) -> None:
        """
        Сериализация закрытого ключа в pem файл
        :param file_path: путь для сохранения
        :param private_key: закрытый ключ
        :return: None
        """
        try:
            with open(file_path, 'wb') as private_out:
                private_out.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
        except Exception as e:
            raise Exception(f"Ошибка записи приватного ключа: {str(e)}")

    @staticmethod
    def deserialization_public_key(file_path: str) -> RSAPublicKey:
        """
        Десериализация открытого ключа из pem файла
        :param file_path: путь к файлу
        :return: открытый ключ
        """
        try:
            with open(file_path, 'rb') as pem_in:
                public_bytes = pem_in.read()
            return load_pem_public_key(public_bytes)
        except Exception as e:
            raise Exception(f"Ошибка чтения публичного ключа: {str(e)}")

    @staticmethod
    def deserialization_private_key(file_path: str) -> RSAPrivateKey:
        """
        Десериализация закрытого ключа из pem файла
        :param file_path: путь к файлу
        :return: закрытый ключ
        """
        try:
            with open(file_path, 'rb') as pem_in:
                private_bytes = pem_in.read()
            return load_pem_private_key(private_bytes, password=None)
        except Exception as e:
            raise Exception(f"Ошибка чтения приватного ключа: {str(e)}")

    @staticmethod
    def serialize_symmetric_key(file_path: str, data: bytes) -> None:
        """
        Сериализация бинарных данных в файл
        :param file_path: путь к файлу
        :param data: бинарные данные
        :return: None
        """
        try:
            with open(file_path, mode='wb') as file:
                file.write(data)
        except Exception as e:
            raise Exception(f"Ошибка записи байтов: {str(e)}")

    @staticmethod
    def deserialize_symmetric_key(file_path: str) -> bytes:
        """
       Десериализация симметричного ключа
        :param file_path: файл с ключом
        :return: ключ в виде байтовой последовательности
        """
        try:
            with open(file_path, 'rb') as file:
                data = file.read()
            return data
        except Exception as e:
            raise Exception(f"Ошибка чтения байтов: {str(e)}")

    @staticmethod
    def write_txt(file_path: str, text: str) -> None:
        """
        Запись текстовых данных в файл
        :param file_path: файл куда записывать текст
        :param text: текст для записи
        :return: None
        """
        try:
            with open(file_path, 'w') as file:
                file.write(text)
        except Exception as e:
            raise Exception(f"Ошибка записи текста: {str(e)}")

    @staticmethod
    def write_json(file_path: str, data: dict) -> None:
        """
        Сохранение в JSON файл
        :param file_path: путь к файлу
        :param data: словарь
        :return: None
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as fp:
                json.dump(data, fp)
        except Exception as e:
            raise Exception(f"Ошибка записи JSON: {str(e)}")

    @staticmethod
    def get_json(file_name: str) -> dict[str, str]:
        """
        Чтение и парсинг JSON файла
        :param file_name: путь к JSON-файлу
        :return: словарь
        """
        try:
            with open(file_name, 'r', encoding='utf-8') as json_file:
                return json.load(json_file)
        except Exception as e:
            raise Exception(f"Ошибка чтения JSON: {str(e)}")

    @staticmethod
    def read_txt(file_name: str) -> str:
        """
        Чтение текстовых данных из файла
        :param file_name: путь к файлу
        :return: текст из файла
        """
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Ошибка чтения файла: {str(e)}.")