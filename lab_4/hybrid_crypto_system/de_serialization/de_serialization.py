from cryptography.hazmat.primitives import serialization

from filehandler import FileHandler


class DeSerialization:
    """Serialization and deserialization operations"""

    @staticmethod
    def serialization_rsa_key(key, key_type):
        """
        Serialization rsa asymmetric key
        :param key_dir: directory to save the key
        :param key: key to save
        :param key_type: type of key - private or public
        :return: serialized key
        """
        match key_type:
            case "private":
                return key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            case "public":
                return key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )

    @staticmethod
    def deserialization_rsa_key(key, key_type):
        """
        Deserialization rsa asymmetric key
        :param key: the key to deserialize
        :param key_type: type of key - private or public
        :return: None
        """
        match key_type:
            case "private":
                return serialization.load_pem_private_key(key, password=None)
            case "public":
                return serialization.load_pem_public_key(key)

    @staticmethod
    def serialization_data(data_dir, data):
        """
        Method serializes the data and saves it to a file
        :param data_dir: directory to save file
        :param data: data to serialize
        :return: None
        """
        FileHandler.save_data(data_dir, data, "wb")

    @staticmethod
    def deserialization_data(data_dir):
        """
        Method deserializes the data from the file
        :param data_dir: directory to serialized file
        :return: deserialized data
        """
        return FileHandler.read_data(data_dir, "rb")
