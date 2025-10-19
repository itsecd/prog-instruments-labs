import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key


class KeyManager:

    def __init__(self):

        self.symmetric_key = None
        self.iv = None
        self.private_key = None
        self.public_key = None

    def validate_key_size(self, key_size):

        if key_size < 32 or key_size > 448 or key_size % 8 != 0:
            raise ValueError(
                "Недопустимый размер ключа. Должен быть 32-448 бит с шагом 8 бит.\n"
                f"Получено: {key_size} бит."
            )

    def generate_symmetric_key(self, key_size=448):

        self.validate_key_size(key_size)
        self.symmetric_key = os.urandom(key_size // 8)
        print(f"Сгенерирован {key_size}-битный симметричный ключ Blowfish")
        return self.symmetric_key

    def generate_asymmetric_keys(self):

        print("Генерация ключей RSA...")
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        print("Ключи RSA успешно сгенерированы")
        return self.private_key, self.public_key

    def load_private_key(self, private_key_path):

        print(f"Загрузка приватного ключа из {private_key_path}...")
        with open(private_key_path, 'rb') as pem_in:
            private_bytes = pem_in.read()
        self.private_key = load_pem_private_key(private_bytes, password=None)
        return self.private_key

    def decrypt_symmetric_key(self, encrypted_sym_key_path):

        print(f"Расшифровка симметричного ключа из {encrypted_sym_key_path}...")
        try:
            with open(encrypted_sym_key_path, 'rb') as key_file:
                encrypted_key = key_file.read()
        except FileNotFoundError:
            print(f"Ошибка: файл ключа не найден - {encrypted_sym_key_path}")
            return None
        except Exception as e:
            print(f"Ошибка чтения файла ключа: {e}")
            return None
        self.symmetric_key = self.private_key.decrypt(
            encrypted_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        print("Симметричный ключ успешно расшифрован")
        return self.symmetric_key