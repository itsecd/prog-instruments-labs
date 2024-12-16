import json

from cryptography.hazmat.primitives import serialization


class Texting:

    def read_json_file(path: str) -> list:
        """Производит чтение json файла, принимает путь, возвращает лист считанных данны"""
        try:
            with open(path, "r", encoding="UTF-8") as file:
                return json.loads(file.read())
        except Exception as e:
            print("Возникла ошибка открытия файла с путями")

    def write_bytes(path: str, key: bytes) -> None:
        """производит запись байтов в файл, принимает путь записи и байты, ничего не возвращает"""
        try:
            with open(path, "wb") as file:
                file.write(key)
        except Exception as e:
            print("Возникла ошибка при записи байтов: ", e)

    def read_bytes(path: str) -> bytes:
        """Производит чтение байтов из файла, принимает путь, возвращает байты"""
        try:
            with open(path, "rb") as file:
                return file.read()
        except Exception as e:
            print("Возникла ошибка при чтении байтов: ", e)

    def read_file(source_file_path: str) -> str:
        """Производит чтение строки из файла, принимает путь, возвращает строку"""
        try:
            with open(source_file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print("Возникла ошибка при чтении файла: ", e)

    def write_file(path: str, text: str) -> None:
        """производит запись строки в файл, принимает путь записи и строку, ничего не возвращает"""
        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
        except Exception as e:
            print("Возникла ошибка при записи файла: ", e)

    def serialize_private(key: dict, path: str) -> None:
        """Производит сериализацию приватного ключа, принимает словарь с асимметричнным ключом и путь, ничего не возвращает"""
        try:
            with open(path, "wb") as file:
                file.write(
                    key["private"].private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )
        except Exception as e:
            print("Возникла ошибка при сиреализации публичного ключа: ", e)

    def serialize_public(key: dict, path: str) -> None:
        """Производит сериализацию публичного ключа, принимает словарь с асимметричнным ключом и путь, ничего не возвращает"""
        try:
            with open(path, "wb") as file:
                file.write(
                    key["public"].public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                )
        except Exception as e:
            print("Возникла ошибка при сиреализацииЭ приватного ключа: ", e)
