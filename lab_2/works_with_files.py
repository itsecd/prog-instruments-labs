import json
import os
import config

from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.hazmat.primitives import serialization


class WorkWithFiles:
    """Class which stores methods to work with files"""
    @staticmethod
    def write_file(path_to_file, content):
        """
        A function for bitwise writing of data to a file
        :param path_to_file: Path to file
        :param content: The data that we write to the file
        """
        print(f"Сохранение данных в файл {path_to_file}...")
        try:
            with open(path_to_file, 'wb') as f:
                f.write(content)
            print(f"||Данные успешно сохранены||")
        except Exception as e:
            print(f"Error: Произошла ошибка при сохранении в файл {e}")
            exit(1)


    @staticmethod
    def read_file(path_to_file):
        """
        A function for bitwise reading of data from a file
        :param path_to_file: Path to file
        :return Content from file
        """
        print(f"Чтение данных из файла {path_to_file}...")
        try:
            with open(path_to_file, 'rb') as f:
                content = f.read()
            print(f"||Данные успешно прочитаны||")
            return content
        except FileNotFoundError:
            print(f"Error: Файл не найден")
            exit(1)
        except Exception as e:
            print(f"Error Произошла ошибка при чтении в файла {e}")
            exit(1)


    @staticmethod
    def read_private_key(path_to_key):
        """
        Reading the private key from a file
        :param path_to_key: Path to private RSA key
        :return: Private key
        """
        print(f"Загрузка закрытого ключа из файла {path_to_key}...")
        try:
            with open(path_to_key, 'rb') as f:
                private_key = load_pem_private_key(f.read(), password=None)
            print("||Приватный ключ загружен!||")
            return private_key
        except FileNotFoundError:
            print(f"Error: Файл приватного ключа не найден по пути {path_to_key}")
            exit(1)
        except Exception as e:
            print(f"Error: Произошла ошибка при загрузки закрытого ключа {e}")
            exit(1)

    @staticmethod
    def read_public_key(path_to_key):
        """
        Reading the public key from a file
        :param path_to_key: Path to public RSA key
        :return: Public key
        """
        print(f"Загрузка открытого ключа из файла {path_to_key}...")
        try:
            with open(path_to_key, 'rb') as f:
                public_key = load_pem_public_key(f.read())
            print("||Открытый ключ загружен||")
            return public_key
        except FileNotFoundError:
            print(f"Error: Файл открытого ключа не найден по пути {path_to_key}")
            exit(1)
        except Exception as e:
            print(f"Error: Произошла ошибка при загрузки открытого ключа {e}")
            exit(1)

    @staticmethod
    def load_config_settings(path_to_file = None):
        """
        Function for loading parameters from a json file
        :param path_to_file: The path to the settings file
        :return: Configuration settings
        """
        print("Загрузка настроек...")
        settings = {
            "initial_file": config.initial_file,
            "encrypted_file": config.encrypted_file,
            "decrypted_file": config.decrypted_file,
            "encrypted_symmetric_key_file": getattr(config, 'encrypted_symmetric_key_file',
                                                    config.symmetric_key),
            "public_key": config.public_key,
            "secret_key": config.secret_key
        }
        if path_to_file:
            print(f"Попытка загрузить настройки из JSON файла {path_to_file} ...")
            if not os.path.exists(path_to_file):
                print(f"Error: JSON файл настроек не найден по пути {path_to_file}")
            else:
                try:
                    with open(path_to_file, 'r', encoding='utf-8') as file:
                        json_settings = json.load(file)
                    settings.update(json_settings)
                    print("||Настройки успешно обновлены из JSON файла||")
                except json.JSONDecodeError:
                    print(f"Error: Неверный формат JSON файла {path_to_file}")
                except Exception as e:
                    print(f"Error: Ошибка при загрузке настроек из файла {e}")

        print("||Настройки загружены!||")
        return settings

    @staticmethod
    def save_public_key(public_key, path_to_save):
        """
        A function for writing an RSA public key to a file
        :param public_key: Public key
        :param path_to_save: Path to save key
        """
        print(f"Сохранение открытого ключа в {path_to_save}...")
        with open(path_to_save, 'wb') as public_out:
            public_out.write(
                public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                        format=serialization.PublicFormat.SubjectPublicKeyInfo))
        print(f"Открытый ключ сохранен в {path_to_save}")

    @staticmethod
    def save_private_key(private_key, path_to_save):
        """
        A function for writing an RSA private key to a file
        :param private_key: Private key
        :param path_to_save: Path to save
        """
        print(f"Сохранение закрытого ключа в {path_to_save}...")
        with open(path_to_save, 'wb') as private_out:
            private_out.write(
                private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                          format=serialization.PrivateFormat.TraditionalOpenSSL,
                                          encryption_algorithm=serialization.NoEncryption()))
        print(f"Закрытый ключ сохранен в {path_to_save}")

    @staticmethod
    def save_encrypt_symmetric_key(encrypted_symmetric_key, settings):
        """
        A function for writing an RSA encrypt symmetric key key to a file
        :param encrypted_symmetric_key: Encrypted symmetric key
        :param settings: An object that stores parameters from a configuration file.
        :return:
        """
        encrypted_sym_key_path = settings['encrypted_symmetric_key_file']
        if not encrypted_sym_key_path:
            encrypted_sym_key_path = settings.get('symmetric_key')
            if not encrypted_sym_key_path:
                print(
                    "Error: Не указан путь для сохранения зашифрованного симметричного ключа в файле конфигурации.")
                exit(1)
        print(f"Сохранение зашиф. симметричного ключа в {settings['encrypted_symmetric_key_file']}...")
        WorkWithFiles.write_file(settings['encrypted_symmetric_key_file'], encrypted_symmetric_key)
        print(f"Зашифрованный симметричный ключ сохранен в {settings['encrypted_symmetric_key_file']}.")
