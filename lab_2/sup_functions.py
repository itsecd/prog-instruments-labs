import os

from cryptography.hazmat.primitives.asymmetric import rsa

from asymmetrical_crypt import Asymmetrical

class SupportFunctions:
    """Class which store support methods"""
    @staticmethod
    def generate_symmetric_key():
        """
        A function that generates a symmetric encryption key
        :return: Generated symmetric encryption key
        """
        print("Генерация симметричного ключа ChaCha20 (256 бит)...")
        symmetric_key = os.urandom(32)
        print("||Симметричный ключ сгенерирован!||")
        return symmetric_key

    @staticmethod
    def generate_rsa_keys(settings):
        """
        A function for generating an RSA key pair
        :param settings: An object that stores parameters from a configuration file.
        """
        print("Генерация пары RSA ключей (размер 2048 бит)...")
        try:
            keys = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            private_key = keys
            public_key = keys.public_key()
            return private_key, public_key
        except Exception as e:
            print(f"Error: Произошла ошибка при генерации или сохранении ключей {e}")
            exit(1)

    @staticmethod
    def generate_keys(settings):
        """
        A function that calls other key generation functions
        :param settings: An object that stores parameters from a configuration file.
        """
        print("\n||Генерация ключей||")
        symmetric_key = SupportFunctions.generate_symmetric_key()
        private_key, public_key = SupportFunctions.generate_rsa_keys(settings)
        encrypted_sym_key_path = settings['encrypted_symmetric_key_file']
        if not encrypted_sym_key_path:
            encrypted_sym_key_path = settings.get('symmetric_key')
            if not encrypted_sym_key_path:
                print("Error: Не указан путь для сохранения зашифрованного симметричного ключа в файле конфигурации.")
                exit(1)
        encrypt_symmetric_key = Asymmetrical.encrypt_symmetric_key(public_key, symmetric_key)
        print("||Генерация ключей завершена успешно!||")
        return public_key, private_key, encrypt_symmetric_key



