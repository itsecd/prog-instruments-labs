from hybrid_crypto_system.asymmetric_crypto.asymmetric_crypto import AsymmetricCrypto
from hybrid_crypto_system.de_serialization.de_serialization import DeSerialization
from filehandler import FileHandler
from hybrid_crypto_system.symmetric_crypto.symmetric_crypto import SymmetricCrypto


class HybridCryptoSystem:
    """Hybrid CryptoSystem class"""

    def __init__(self, length=192):
        """
        Initializing the system
        :param length: key length, default value=192 bits
        """
        self.__key_length = length

    def generate_keys(
        self,
        encrypted_symmetric_key_dir: str,
        private_key_dir: str,
        public_key_dir: str,
    ):
        """
        Key generation method
        :param encrypted_symmetric_key_dir: directory to save encrypted symmetric key
        :param public_key_dir: directory to save public asymmetric key
        :param private_key_dir: directory to save private asymmetric key
        :return None
        """
        private_key, public_key = AsymmetricCrypto.generate_keys(self.__key_length)
        symmetric_key = SymmetricCrypto.generate_key(self.__key_length)

        s_private_key = DeSerialization.serialization_rsa_key(private_key, "private")
        s_public_key = DeSerialization.serialization_rsa_key(public_key, "public")
        encrypted_symmetric_key = AsymmetricCrypto.encrypt_symmetric_key(
            symmetric_key, public_key
        )

        DeSerialization.serialization_data(
            encrypted_symmetric_key_dir, encrypted_symmetric_key
        )
        DeSerialization.serialization_data(private_key_dir, s_private_key)
        DeSerialization.serialization_data(public_key_dir, s_public_key)

    def encrypt_data(
        self,
        plain_text_dir: str,
        private_key_dir: str,
        encrypted_symmetric_key_dir: str,
        encrypted_data_dir: str,
    ):
        """
        Data encryption method
        :param plain_text_dir: directory to file with text to encrypt
        :param private_key_dir: directory to private asymmetric key
        :param encrypted_symmetric_key_dir: directory to symmetric key
        :param encrypted_data_dir: directory to save encrypted data
        :return: None
        """
        if not (plain_text := FileHandler.read_data(plain_text_dir, "rb")):
            raise ValueError("Text file must not be empty")
        if not (private_bytes := FileHandler.read_data(private_key_dir, "rb")):
            raise ValueError("Private key must not be empty")
        if not (
            encrypted_symmetric_key := FileHandler.read_data(
                encrypted_symmetric_key_dir, "rb"
            )
        ):
            raise ValueError("Encrypted symmetric key must not be empty")

        encrypted_data = SymmetricCrypto.encrypt_data(
            plain_text, private_bytes, encrypted_symmetric_key
        )
        # FileHandler.save_data(encrypted_data_dir, encrypted_data, "wb")
        DeSerialization.serialization_data(encrypted_data_dir, encrypted_data)

    def decrypt_data(
        self,
        encrypted_text_dir: str,
        private_key_dir: str,
        encrypted_symmetric_key_dir: str,
        decrypted_text_dir: str,
    ):
        """
        Decryption data method
        :param encrypted_text_dir: directory to file with text to decrypt
        :param private_key_dir: directory to private asymmetric key
        :param encrypted_symmetric_key_dir: directory to symmetric key
        :param decrypted_text_dir: irectory to save decrypted data
        :return: None
        """
        if not (encrypted_text := FileHandler.read_data(encrypted_text_dir, "rb")):
            raise ValueError("Text file must not be empty")
        if not (private_bytes := FileHandler.read_data(private_key_dir, "rb")):
            raise ValueError("Private key must not be empty")
        if not (
            encrypted_symmetric_key := FileHandler.read_data(
                encrypted_symmetric_key_dir, "rb"
            )
        ):
            raise ValueError("Encrypted symmetric key must not be empty")

        decrypted_data = SymmetricCrypto.decrypt_data(
            encrypted_text, private_bytes, encrypted_symmetric_key
        )
        FileHandler.save_data(decrypted_text_dir, decrypted_data, "wb")
