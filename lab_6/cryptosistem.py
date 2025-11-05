from symmetric import Symmetric
from lab_6.asymmetric import Asymmetric


class CryptoSistem:
    def __init__(self, key_len):
        self._key_len = key_len
        self._iv = Symmetric.iv()

    def generate_hybrid_keys(self) -> tuple:
        """
        Runs 1st scenario: generate keys
        :return: tuple (encrypted_symmetric_key, public_key, private_key)
        """
        symmetric_key = Symmetric.generate_symmetric_key(self._key_len)
        private_key, public_key = Asymmetric.generate_asymmetric_keys()
        c_symmetric_key = Asymmetric.encrypt_symmetric_key(public_key, symmetric_key)

        return c_symmetric_key, public_key, private_key

    def encrypt_data(self, plain_text: bytes, private_key, c_symmetric_key: bytes) -> bytes:
        """
        Runs 2nd scenario: encrypt text
        :param plain_text: plain text to encrypt
        :param private_key: private asymmetric key
        :param c_symmetric_key: encrypted symmetric key
        :return: encrypted text
        """
        symmetric_key = Asymmetric.decrypt_symmetric_key(private_key, c_symmetric_key)
        encrypted_text = Symmetric.encrypt_text(plain_text, symmetric_key, self._iv)

        return encrypted_text

    def decrypt_data(self, encrypted_text: bytes, private_key, c_symmetric_key: bytes) -> str:
        """
        Runs 3rd scenario: decrypt text
        :param encrypted_text: encrypted text
        :param private_key: private asymmetric key
        :param c_symmetric_key: encrypted symmetric key
        :return: decrypted text as string
        """
        symmetric_key = Asymmetric.decrypt_symmetric_key(private_key, c_symmetric_key)
        decrypted_bytes = Symmetric.decrypt_text(encrypted_text, symmetric_key, self._iv)
        decrypted_text = decrypted_bytes.decode('utf-8')

        return decrypted_text