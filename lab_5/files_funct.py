import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    load_pem_private_key,
)


class FilesFunct:
    """
    This class work with files
    """
    def read_text_from_file(file_path: str) -> str:
        """
        This function reads the file format .txt
        Arguments:
            file_path: location the .txt file whose contents you want to find out
        Returns:
            text: content .txt file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                return text
        except FileNotFoundError:
            print("Файл не найден.")
        except Exception as e:
            print(f"Произошла ошибка при чтении файла .txt: {e}")


    def write_to_txt_file(text: str, file_name: str) -> None:
        """
        This function writes data to a .txt file
        Arguments:
            text: the data that we write to the file
            file_name: location the .txt file in which we write the data
        """
        try:
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(text)
            print(f"Текст успешно записан в файл .txt")
        except Exception as e:
            print(f"Произошла ошибка при записи в файл .txt: {e}")


    def read_json_file(file_name: str) -> dict[str, str]:
        """
        This function reads the file format .json
        Arguments:
            file_name: location the .json file whose contents you want to find out
        Returns:
            data: dictionary of keys
        """
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Произошла ошибка при чтении файла .json: {e}")
            return None


    def read_bytes_from_file(file_path: str) -> bytes:
        """
        This function reads the file with bytes
        Arguments:
            file_path: location the file whose contents you want to find out
        Returns:
            num_bytes: content file
        """
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except Exception as e:
            print(f"Произошла ошибка при чтении bytes: {e}")
            return None


    def write_bytes_to_file(file_path: str, data: bytes) -> None:
        """
        This function writes bytes to file
        Arguments:
            file_path: the bytes that we write to the file
            data: location the file in which we write bytes
        """
        try:
            with open(file_path, "wb") as file:
                file.write(data)
            print(f"Успешно записано в файл .txt")
        except Exception as e:
            print(f"Произошла ошибка при записи bytes в файл: {e}")


    def deserialization_rsa_public_key(public_pem: str) -> rsa.RSAPublicKey:
        """
        Deserialize an RSA public key from a PEM file.
        Arguments:
            public_pem (str): Path to the PEM file containing the public key.
        Returns:
            rsa.RSAPublicKey: The deserialized RSA public key.
        """
        try:
            with open(public_pem, "rb") as pem_in:
                public_bytes = pem_in.read()
                d_public_key = load_pem_public_key(public_bytes)
                print(f"Открытый ключ успешно десериализирован в файл pem" )
                return d_public_key
        except Exception as e:
            print(f"Произошла ошибка при десериализации открытого ключа: {e}")


    def serialization_rsa_public_key(public_key: rsa.RSAPublicKey, public_pem: str) -> None:
        """
        Serialize an RSA public key to a PEM file.
        Arguments:
            public_key (rsa.RSAPublicKey): The RSA public key to be serialized.
            public_pem (str): Path to save the serialized public key as a PEM file.
        Returns:
            None
        """
        try:
            with open(public_pem, "wb") as public_out:
                public_out.write(
                    public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                )
            print(f"Открытый ключ успешно сериализирован в файл")
        except Exception as e:
            print(f"Произошла ошибка при сериализации открытого ключа в файл pem: {e}")


    def deserialization_rsa_private_key(private_pem: str) -> rsa.RSAPrivateKey:
        """
        Deserialize an RSA private key from a PEM file.
        Arguments:
            private_pem (str): Path to the PEM file containing the private key.
        Returns:
            rsa.RSAPrivateKey: The deserialized RSA private key.
        """
        try:
            with open(private_pem, "rb") as pem_in:
                private_bytes = pem_in.read()
                d_private_key = load_pem_private_key(
                    private_bytes,
                    password=None,
                )
            print(f"Закрытый ключ успешно десериализирован в файл pem")
            return d_private_key
        except Exception as e:
            print(f"Произошла ошибка при десериализации закрытого ключа: {e}")


    def serialization_rsa_private_key(
        private_key: rsa.RSAPrivateKey, private_pem: str
    ) -> None:
        """
        Serialize an RSA private key to a PEM file.
        Arguments:
            private_key (rsa.RSAPublicKey): The RSA private key to be serialized.
            private_pem (str): Path to save the serialized private key as a PEM file.
        Returns:
            None
        """
        try:
            with open(private_pem, "wb") as private_out:
                private_out.write(
                    private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )
            print(f"Закрытый ключ успешно сериализирован в файл pem")
        except Exception as e:
            print(f"Произошла ошибка при сериализации закрытого ключа в файл: {e}")


    def serial_sym_key(file_name: str, key: bytes) -> None:
        """
        Serialize a symmetric key to a file.
        Arguments:
            file_name (str): Path to save the serialized symmetric key.
            key (bytes): The symmetric key to be serialized.
        Returns:
            None
        """
        try:
            with open(file_name, "wb") as key_file:
                key_file.write(key)
            print(f"Симметричный ключ успешно сериализирован в файл .txt")
        except Exception as e:
            print(f"Произошла ошибка при сериализации симметричного ключа в файл: {e}")


    def deserial_sym_key(file_name: str) -> bytes:
        """
        Deserialize a symmetric key from a file.
        Arguments:
            file_name (str): Path to the file containing the serialized symmetric key.
        Returns:
            bytes: The deserialized symmetric key.
        """
        try:
            with open(file_name, mode="rb") as key_file:
                content = key_file.read()
                print(f"Симметричный ключ успешно десериализирован в файл .txt")
                return content
        except Exception as e:
            print(f"Произошла ошибка при десериализации симметричного ключа: {e}")
            return None
 