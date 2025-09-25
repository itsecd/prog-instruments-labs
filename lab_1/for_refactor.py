import os
import json
import argparse
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_config()
        return cls._instance

    def init_config(self):
        self.config_path = 'settings.json'
        self.default_config = {
            'initial_file': 'texts/original.txt',
            'encrypted_file': 'texts/encrypted.bin',
            'decrypted_file': 'texts/decrypted.txt',
            'symmetric_key': 'keys/sym_key.enc',
            'public_key': 'keys/public.pem',
            'private_key': 'keys/private.pem'
        }
        self.ensure_config_exists()

    def ensure_config_exists(self):
        if not os.path.exists(self.config_path):
            self.create_directories()
            self.save_config(self.default_config)
        else:
            with open(self.config_path, 'r') as f:
                existing_config = json.load(f)
                for key in self.default_config:
                    if key not in existing_config:
                        existing_config[key] = self.default_config[key]
                self.save_config(existing_config)

    def create_directories(self):
        os.makedirs('texts', exist_ok=True)
        os.makedirs('keys', exist_ok=True)

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save_config(self, config):
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def get_setting(self, key):
        config = self.load_config()
        return config.get(key)


class FileManager:
    def __init__(self, texts_dir='texts', keys_dir='keys'):
        self._texts_dir = texts_dir
        self._keys_dir = keys_dir
        self.ensure_dirs_exist()

    def ensure_dirs_exist(self):
        if not os.path.exists(self._texts_dir):
            os.makedirs(self._texts_dir)
        if not os.path.exists(self._keys_dir):
            os.makedirs(self._keys_dir)

    def read_file(self, file_path):
        with open(file_path, 'rb') as f:
            return f.read()

    def write_file(self, file_path, data):
        with open(file_path, 'wb') as f:
            f.write(data)

    @property
    def texts_dir(self):
        return self._texts_dir

    @property
    def keys_dir(self):
        return self._keys_dir


class RsaCipher:
    def __init__(self, key_size=2048):
        self.key_size = key_size

    def generate_keys(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def serialize_private_key(self, private_key, file_path):
        with open(file_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

    def serialize_public_key(self, public_key, file_path):
        with open(file_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

    def load_private_key(self, file_path):
        with open(file_path, 'rb') as f:
            return serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )

    def load_public_key(self, file_path):
        with open(file_path, 'rb') as f:
            return serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )

    def encrypt(self, plaintext, public_key):
        return public_key.encrypt(
            plaintext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def decrypt(self, ciphertext, private_key):
        return private_key.decrypt(
            ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )


class BlowfishCipher:
    def __init__(self, key_length=448):
        if key_length < 32 or key_length > 448 or key_length % 8 != 0:
            raise ValueError("Blowfish Ключ должен быть от 32 до 448, делиться на 8")
        self.key_length = key_length
        self.block_size = 64

    def generate_key(self):
        return os.urandom(self.key_length // 8)

    def encrypt(self, plaintext, key):
        init_vec = os.urandom(8)
        padder = padding.PKCS7(self.block_size).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(
            algorithms.Blowfish(key),
            modes.CBC(init_vec),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return init_vec + ciphertext

    def decrypt(self, ciphertext, key):
        init_vec = ciphertext[:8]
        ciphertext = ciphertext[8:]
        cipher = Cipher(
            algorithms.Blowfish(key),
            modes.CBC(init_vec),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = padding.PKCS7(self.block_size).unpadder()
        decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
        return decrypted


class HybridCryptoSystem:
    def __init__(self, symmetric_key_length=448):
        self.blowfish = BlowfishCipher(symmetric_key_length)
        self.rsa = RsaCipher()

    def generate_keys(self):
        symmetric_key = self.blowfish.generate_key()
        private_key, public_key = self.rsa.generate_keys()
        return symmetric_key, private_key, public_key

    def encrypt_symmetric_key(self, symmetric_key, public_key):
        return self.rsa.encrypt(symmetric_key, public_key)

    def decrypt_symmetric_key(self, encrypted_key, private_key):
        return self.rsa.decrypt(encrypted_key, private_key)

    def encrypt_file(self, plaintext, symmetric_key):
        return self.blowfish.encrypt(plaintext, symmetric_key)

    def decrypt_file(self, ciphertext, symmetric_key):
        return self.blowfish.decrypt(ciphertext, symmetric_key)


def parse_arguments():
    description = (
        'Гибридная криптосистема RSA + Blowfish, которая обеспечивает '
        'безопасную передачу данных через сочетание асимметричного и '
        'симметричного шифрования'
    )
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen', '--generation', action='store_true',
                       help='Запускает режим генерации ключей')
    group.add_argument('-enc', '--encryption', action='store_true',
                       help='Запускает режим шифрования')
    group.add_argument('-dec', '--decryption', action='store_true',
                       help='Запускает режим дешифрования')
    key_length_help = (
        'Длина ключа Blowfish (32-448 бит, шаг 8) - только для генерации'
    )
    parser.add_argument('--key-length', type=int, default=448,
                        help=key_length_help)
    args = parser.parse_args()
    return args


def main():
    config = ConfigManager()
    file_manager = FileManager()
    args = parse_arguments()
    crypto = HybridCryptoSystem()
    try:
        if args.generation:
            print("Генерация ключей...")
            crypto = HybridCryptoSystem(args.key_length)
            symmetric_key, private_key, public_key = crypto.generate_keys()
            crypto.rsa.serialize_private_key(private_key, config.get_setting('private_key'))
            crypto.rsa.serialize_public_key(public_key, config.get_setting('public_key'))
            encrypted_key = crypto.encrypt_symmetric_key(symmetric_key, public_key)
            file_manager.write_file(config.get_setting('symmetric_key'), encrypted_key)
            print("Ключи успешно сгенерированы (в папку keys):")
            print(f"- Симметричный ключ (зашифрованный): sym_key.enc")
            print(f"- Открытый ключ RSA: public.pem")
            print(f"- Закрытый ключ RSA: private.pem")
            print(f"Длина ключа Blowfish: {args.key_length} бит")

        elif args.encryption:
            print("Шифрование файла...")
            private_key = crypto.rsa.load_private_key(config.get_setting('private_key'))
            encrypted_key = file_manager.read_file(config.get_setting('symmetric_key'))
            symmetric_key = crypto.decrypt_symmetric_key(encrypted_key, private_key)
            plaintext = file_manager.read_file(config.get_setting('initial_file'))
            ciphertext = crypto.encrypt_file(plaintext, symmetric_key)
            file_manager.write_file(config.get_setting('encrypted_file'), ciphertext)
            print(f"Файл успешно зашифрован: {config.get_setting('encrypted_file')}")

        elif args.decryption:
            print("Дешифрование файла...")
            private_key = crypto.rsa.load_private_key(config.get_setting('private_key'))
            encrypted_key = file_manager.read_file(config.get_setting('symmetric_key'))
            symmetric_key = crypto.decrypt_symmetric_key(encrypted_key, private_key)
            ciphertext = file_manager.read_file(config.get_setting('encrypted_file'))
            plaintext = crypto.decrypt_file(ciphertext, symmetric_key)
            file_manager.write_file(config.get_setting('decrypted_file'), plaintext)
            print(f"Файл успешно расшифрован: {config.get_setting('decrypted_file')}")

    except FileNotFoundError:
        print("Ошибка: файл с текстом не найден")
    except UnicodeDecodeError:
        print("Ошибка: проблема с кодировкой файла")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        exit(1)


if __name__ == "__main__":
    main()