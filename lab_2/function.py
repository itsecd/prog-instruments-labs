import os

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend


class SymmetricEncryption:
    @staticmethod
    def generate_sm4_key() -> bytes:
        """
        Generates a random key for SM4 encryption
        :return:random key
        """
        return os.urandom(16)

    @staticmethod
    def sm4_encrypt(key: bytes, text: bytes) -> bytes:
        """
        Encrypts text using the SM4 algorithm in CBC mode
        :param key:the encryption key is 16 bytes (128 bits) long
        :param text:text for encryption in the form of bytes
        :return:encrypted text
        """
        if len(key) != 16:
            raise ValueError("The key must be 16 bytes (128 bits)")
        iv = os.urandom(16)
        cipher = Cipher(algorithms.SM4(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        pad_length = 16 - (len(text) % 16)
        padded_plaintext = text + bytes([pad_length] * pad_length)
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        return iv + ciphertext

    @staticmethod
    def sm4_decrypt(key: bytes, ciphertext: bytes) -> bytes:
        """
        Decrypts cipher text using the SM4 algorithm in CBC mode
        :param key:the decryption key is 16 bytes (128 bits) long
        :param ciphertext:cipher text
        :return:decrypted text
        """
        if len(key) != 16:
            raise ValueError("The key must be 16 bytes (128 bits)")
        iv = ciphertext[:16]
        actual_ciphertext = ciphertext[16:]
        cipher = Cipher(algorithms.SM4(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_text = decryptor.update(actual_ciphertext) + decryptor.finalize()
        pad_length = padded_text[-1]
        text = padded_text[:-pad_length]
        return text


class AsymmetricEncryption:
    @staticmethod
    def generate_rsa_keys() -> tuple:
        """
        Generates an RSA key pair (private and public)
        :return:RSA private and public keys
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def serialize_private_key(private_key: rsa.RSAPrivateKey, filename: str) -> None:
        """
        Serializes the RSA private key to a file
        :param private_key: the RSA private key
        :param filename:the name of the file to save the key to
        :return:None
        """
        with open(filename, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

    @staticmethod
    def serialize_public_key(public_key: rsa.RSAPublicKey, filename: str) -> None:
        """
        Serializes the RSA public key to a file
        :param public_key: the RSA public key
        :param filename:the name of the file to save the key to
        :return:None
        """
        with open(filename, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

    @staticmethod
    def load_private_key(filename: str) -> rsa.RSAPrivateKey:
        """
        Downloads the RSA private key from a file
        :param filename:the file name for uploading the key
        :return:the RSA private key
        """
        with open(filename, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        return private_key

    @staticmethod
    def load_public_key(filename: str) -> rsa.RSAPublicKey:
        """
        Downloads the RSA public key from a file
        :param filename:the file name for uploading the key
        :return:the RSA public key
        """
        with open(filename, 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        return public_key

    @staticmethod
    def encrypt_symmetric_key(symmetric_key: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        """
        Encrypts the symmetric key using the RSA public key
        :param symmetric_key:symmetric encryption key
        :param public_key:RSA public key for encryption
        :return: encrypted symmetric key
        """
        ciphertext = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext

    @staticmethod
    def decrypt_symmetric_key(ciphertext: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """
        Decrypts a symmetric key using an RSA private key
        :param ciphertext:encrypted symmetric key
        :param private_key:RSA private key for decryption
        :return:decrypted symmetric key
        """
        symmetric_key = private_key.decrypt(ciphertext, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                     algorithm=hashes.SHA256(), label=None))
        return symmetric_key


class FileHandler:
    @staticmethod
    def read_txt_file(file_path: str) -> bytes:
        """
        A function for reading a text file as bytes.
        :param file_path: path to the text file
        :return: text file as bytes
        """
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except Exception as e:
            print(f"Error: {e}")
            return b''

    @staticmethod
    def write_txt_file(data: bytes, file_path: str) -> None:
        """
        A function for writing bytes to a file.
        :param data: data to enter into the file as bytes
        :param file_path: the path to the file to save the data
        :return: None
        """
        try:
            with open(file_path, 'wb') as file:
                file.write(data)
        except Exception as e:
            print(f"Error: {e}")