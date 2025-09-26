import const

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from file_work import FileWork


class RSA:
    @staticmethod
    def generate_rsa_keys(key_size=2048):
        """
        Генерация пары RSA ключей
        Параметры:
        key_size (int): размер ключа в битах (по умолчанию 2048)
        Возвращает:
        tuple: (приватный ключ, публичный ключ)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def save_rsa_keys(private_key, public_key,
                      priv_path=const.PATH_TO_PRIVATE_KEY,
                      pub_path=const.PATH_TO_PUBLIC_KEY):
        """
        Сохранение RSA ключей в файлы
        Параметры:
        private_key: закрытый ключ RSA
        public_key: открытый ключ RSA
        priv_path (str): путь для сохранения закрытого ключа
        pub_path (str): путь для сохранения открытого ключа
        Возвращает:
        tuple: (сериализованный закрытый ключ, сериализованный открытый ключ)
        """
        # Сериализация закрытого ключа
        priv_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )

        # Сериализация открытого ключа
        pub_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Запись в файлы
        FileWork.write_file(priv_path, priv_pem)
        FileWork.write_file(pub_path, pub_pem)

        return priv_pem, pub_pem

    @staticmethod
    def load_rsa_public_key(pub_path=const.PATH_TO_PUBLIC_KEY):
        """
        Загрузка открытого ключа RSA из файла
        Параметры:
        pub_path (str): путь к файлу с открытым ключом
        Возвращает:
        PublicKey: объект открытого ключа или None при ошибке
        """
        pub_pem = FileWork.read_file(pub_path)
        if not pub_pem:
            return None
        return serialization.load_pem_public_key(pub_pem, backend=default_backend())

    @staticmethod
    def load_rsa_private_key(priv_path=const.PATH_TO_PRIVATE_KEY):
        """
        Загрузка закрытого ключа RSA из файла
        Параметры:
        priv_path (str): путь к файлу с закрытым ключом
        Возвращает:
        PrivateKey: объект закрытого ключа или None при ошибке
        """
        priv_pem = FileWork.read_file(priv_path)
        if not priv_pem:
            return None
        return serialization.load_pem_private_key(priv_pem, password=None, backend=default_backend())

    @staticmethod
    def encrypt_rsa(public_key, plaintext):
        """
        Шифрование данных с помощью RSA-OAEP
        Параметры:
        public_key: открытый ключ RSA
        plaintext (bytes): данные для шифрования
        Возвращает:
        bytes: зашифрованные данные
        """
        return public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    @staticmethod
    def decrypt_rsa(private_key, ciphertext):
        """
           Дешифрование данных с помощью RSA-OAEP
           Параметры:
           private_key: закрытый ключ RSA
           ciphertext (bytes): зашифрованные данные
           Возвращает:
           bytes: расшифрованные данные
           """
        return private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
