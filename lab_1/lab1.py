import argparse
import os
import json
from typing import Any, Tuple, Dict

from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key, load_pem_private_key
)
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey, RSAPrivateKey


class RSACrypto:
    """Обработчик RSA шифрования для защиты ключей."""

    @staticmethod
    def rsa_encrypt(public_key: RSAPublicKey, data: bytes) -> bytes:
        """Шифрует данные RSA публичным ключом.

        Использует OAEP padding с SHA256 для надежности.

        Args:
            public_key: Публичный ключ для шифрования
            data: Данные для шифрования

        Returns:
            Зашифрованные данные

        Raises:
            ValueError: Если данные слишком большие для RSA
            Exception: При ошибках шифрования
        """
        try:
            print("Начинаем RSA шифрование...")

            encrypted = public_key.encrypt(
                data,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            print("RSA шифрование завершено")
            return encrypted

        except ValueError as e:
            print(f"Слишком много данных для RSA: {e}")
            raise
        except Exception as e:
            print(f"Ошибка RSA шифрования: {e}")
            raise

    @staticmethod
    def rsa_decrypt(private_key: RSAPrivateKey, encrypted_data: bytes) -> bytes:
        """Расшифровывает данные RSA приватным ключом.

        Args:
            private_key: Приватный ключ для расшифровки
            encrypted_data: Зашифрованные данные

        Returns:
            Расшифрованные данные

        Raises:
            Exception: При ошибках расшифровки
        """
        try:
            print("Начинаем RSA дешифрование...")

            decrypted = private_key.decrypt(
                encrypted_data,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            print("RSA дешифрование завершено")
            return decrypted

        except Exception as e:
            print(f"Ошибка RSA дешифрования: {e}")
            raise


class AESCrypto:
    """AES шифрование с CBC режимом для данных."""

    @staticmethod
    def create_iv() -> bytes:
        """Создает случайный вектор инициализации."""
        return os.urandom(16)

    @staticmethod
    def create_aes_key(key_size: int = 32) -> bytes:
        """Генерирует AES ключ нужного размера.

        Args:
            key_size: Размер ключа в байтах (16, 24, 32)

        Returns:
            Сгенерированный ключ

        Raises:
            ValueError: При недопустимом размере ключа
        """
        if key_size not in [16, 24, 32]:
            raise ValueError("Допустимые размеры ключа: 16, 24, 32 байта")
        return os.urandom(key_size)

    @staticmethod
    def encrypt_data(data: bytes, key: bytes, iv: bytes = None) -> Tuple[bytes, bytes]:
        """Шифрует данные AES-256 в режиме CBC.

        Args:
            data: Данные для шифрования
            key: Ключ для шифрования
            iv: Вектор инициализации (опционально)

        Returns:
            Кортеж (зашифрованные данные, вектор инициализации)

        Raises:
            ValueError: При неверном размере ключа
            Exception: При ошибках шифрования
        """
        try:
            if iv is None:
                iv = AESCrypto.create_iv()

            if len(key) not in [16, 24, 32]:
                raise ValueError("Неверный размер AES ключа")

            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )

            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()

            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            print("AES шифрование успешно")
            return ciphertext, iv

        except Exception as e:
            print(f"Ошибка AES шифрования: {e}")
            raise

    @staticmethod
    def decrypt_data(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
        """Расшифровывает данные AES.

        Args:
            ciphertext: Зашифрованные данные
            key: Ключ для расшифровки
            iv: Вектор инициализации

        Returns:
            Расшифрованные данные

        Raises:
            Exception: При ошибках расшифровки
        """
        try:
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )

            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()

            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_data) + unpadder.finalize()

            print("AES дешифрование успешно")
            return plaintext

        except Exception as e:
            print(f"Ошибка AES дешифрования: {e}")
            raise


class FileManager:
    """Управление файловыми операциями."""

    @staticmethod
    def read_file(filename: str) -> bytes:
        """Читает файл и возвращает его содержимое.

        Args:
            filename: Путь к файлу

        Returns:
            Содержимое файла в виде байтов

        Raises:
            FileNotFoundError: Если файл не найден
            Exception: При других ошибках чтения
        """
        try:
            with open(filename, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Файл не найден: {filename}")
            raise
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            raise

    @staticmethod
    def write_file(filename: str, data: bytes) -> None:
        """Записывает данные в файл.

        Args:
            filename: Путь к файлу
            data: Данные для записи

        Raises:
            Exception: При ошибках записи
        """
        try:
            with open(filename, 'wb') as f:
                f.write(data)
            print(f"Данные сохранены в: {filename}")
        except Exception as e:
            print(f"Ошибка записи файла: {e}")
            raise

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Загружает конфигурацию из JSON файла.

        Args:
            config_path: Путь к файлу конфигурации

        Returns:
            Загруженная конфигурация в виде словаря

        Raises:
            Exception: При ошибках загрузки
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            raise

    @staticmethod
    def save_public_key(key: RSAPublicKey, filename: str) -> None:
        """Сохраняет публичный ключ в PEM формате.

        Args:
            key: Публичный ключ для сохранения
            filename: Путь к файлу

        Raises:
            Exception: При ошибках сохранения
        """
        try:
            pem_data = key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            FileManager.write_file(filename, pem_data)
        except Exception as e:
            print(f"Ошибка сохранения публичного ключа: {e}")
            raise

    @staticmethod
    def save_private_key(key: RSAPrivateKey, filename: str) -> None:
        """Сохраняет приватный ключ в PEM формате.

        Args:
            key: Приватный ключ для сохранения
            filename: Путь к файлу

        Raises:
            Exception: При ошибках сохранения
        """
        try:
            pem_data = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )
            FileManager.write_file(filename, pem_data)
        except Exception as e:
            print(f"Ошибка сохранения приватного ключа: {e}")
            raise

    @staticmethod
    def load_public_key(filename: str) -> RSAPublicKey:
        """Загружает публичный ключ из PEM файла.

        Args:
            filename: Путь к файлу с публичным ключом

        Returns:
            Загруженный публичный ключ

        Raises:
            Exception: При ошибках загрузки
        """
        try:
            key_data = FileManager.read_file(filename)
            return load_pem_public_key(key_data)
        except Exception as e:
            print(f"Ошибка загрузки публичного ключа: {e}")
            raise

    @staticmethod
    def load_private_key(filename: str) -> RSAPrivateKey:
        """Загружает приватный ключ из PEM файла.

        Args:
            filename: Путь к файлу с приватным ключом

        Returns:
            Загруженный приватный ключ

        Raises:
            Exception: При ошибках загрузки
        """
        try:
            key_data = FileManager.read_file(filename)
            return load_pem_private_key(key_data, password=None)
        except Exception as e:
            print(f"Ошибка загрузки приватного ключа: {e}")
            raise


class KeyManager:
    """Управление генерацией и защитой ключей."""

    @staticmethod
    def generate_key_pair(key_size: int = 2048) -> Tuple[RSAPublicKey, RSAPrivateKey]:
        """Создает пару RSA ключей.

        Args:
            key_size: Размер ключа в битах

        Returns:
            Кортеж (публичный ключ, приватный ключ)
        """
        print("Генерируем RSA ключи...")

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        public_key = private_key.public_key()

        print(f"RSA-{key_size} ключи созданы")
        return public_key, private_key

    @staticmethod
    def generate_aes_key() -> Tuple[bytes, bytes]:
        """Создает AES ключ и вектор инициализации.

        Returns:
            Кортеж (AES ключ, вектор инициализации)
        """
        print("Генерируем AES ключ...")

        aes_key = AESCrypto.create_aes_key(32)  # AES-256
        iv = AESCrypto.create_iv()

        print("AES ключ и IV созданы")
        return aes_key, iv

    @staticmethod
    def protect_aes_key(public_key: RSAPublicKey, aes_key: bytes) -> bytes:
        """Защищает AES ключ RSA шифрованием.

        Args:
            public_key: Публичный ключ для шифрования
            aes_key: AES ключ для защиты

        Returns:
            Зашифрованный AES ключ
        """
        encrypted_key = RSACrypto.rsa_encrypt(public_key, aes_key)
        print("AES ключ защищен")

        return encrypted_key


class CryptoSystem:
    """Главная система гибридного шифрования."""

    def __init__(self, config_path: str) -> None:
        """Инициализирует систему шифрования с конфигурацией.

        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config = FileManager.load_config(config_path)

    def generate_keys(self) -> None:
        """Генерирует все необходимые ключи и сохраняет их."""
        print("\n" + "=" * 50)
        print("ЗАПУСК ГЕНЕРАЦИИ КЛЮЧЕЙ")
        print("=" * 50)

        public_key, private_key = KeyManager.generate_key_pair()

        aes_key, iv = KeyManager.generate_aes_key()

        encrypted_aes_key = KeyManager.protect_aes_key(public_key, aes_key)

        FileManager.save_public_key(
            public_key, self.config['public_key_file']
        )
        FileManager.save_private_key(
            private_key, self.config['private_key_file']
        )
        FileManager.write_file(
            self.config['encrypted_key_file'], encrypted_aes_key
        )
        FileManager.write_file(self.config['iv_file'], iv)

        print("Все ключи успешно созданы и сохранены!")

    def encrypt_file(self) -> None:
        """Шифрует исходный файл используя гибридное шифрование."""
        print("\n" + "=" * 50)
        print("ЗАПУСК ШИФРОВАНИЯ")
        print("=" * 50)

        private_key = FileManager.load_private_key(
            self.config['private_key_file']
        )
        encrypted_aes_key = FileManager.read_file(
            self.config['encrypted_key_file']
        )
        iv = FileManager.read_file(self.config['iv_file'])

        print("Восстанавливаем AES ключ...")
        aes_key = RSACrypto.rsa_decrypt(private_key, encrypted_aes_key)

        print("Читаем данные для шифрования...")
        plaintext = FileManager.read_file(self.config['source_file'])

        print("Шифруем данные AES...")
        ciphertext, _ = AESCrypto.encrypt_data(plaintext, aes_key, iv)

        FileManager.write_file(self.config['encrypted_file'], ciphertext)

        print("Файл успешно зашифрован!")

    def decrypt_file(self) -> None:
        """Расшифровывает файл используя гибридное шифрование."""
        print("\n" + "=" * 50)
        print("ЗАПУСК ДЕШИФРОВАНИЯ")
        print("=" * 50)

        private_key = FileManager.load_private_key(
            self.config['private_key_file']
        )
        encrypted_aes_key = FileManager.read_file(
            self.config['encrypted_key_file']
        )
        iv = FileManager.read_file(self.config['iv_file'])

        print("Восстанавливаем AES ключ...")
        aes_key = RSACrypto.rsa_decrypt(private_key, encrypted_aes_key)

        print("Читаем зашифрованные данные...")
        ciphertext = FileManager.read_file(self.config['encrypted_file'])

        print("Дешифруем данные AES...")
        plaintext = AESCrypto.decrypt_data(ciphertext, aes_key, iv)

        FileManager.write_file(self.config['decrypted_file'], plaintext)

        print("Файл успешно расшифрован!")


def main() -> int:
    """Главная функция для запуска системы шифрования.

    Returns:
        Код завершения программы (0 - успех, 1 - ошибка)
    """
    parser = argparse.ArgumentParser(
        description="Система гибридного шифрования"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen', action='store_true', help='Генерация ключей')
    group.add_argument('-enc', action='store_true', help='Шифрование файла')
    group.add_argument('-dec', action='store_true', help='Дешифрование файла')

    parser.add_argument(
        '-c', '--config', required=True, help='Путь к конфигурации'
    )

    args = parser.parse_args()

    try:
        system = CryptoSystem(args.config)

        if args.gen:
            system.generate_keys()
        elif args.enc:
            system.encrypt_file()
        elif args.dec:
            system.decrypt_file()

    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())