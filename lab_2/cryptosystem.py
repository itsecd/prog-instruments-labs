import os

from asymmetric import AsymmetricCipher
from file_operations import FileManager
from symmetric import SymmetricCipher
from serialize import SerializeKeys


class CryptoSystem:
    @staticmethod
    def generate_keys(sym_key_length, enc_key_path, pub_key_path, private_key_path):
        """
        Generates and saves all necessary cryptographic keys

        :param sym_key_length: Length of symmetric key in bits (128, 192 or 256)
        :param enc_key_path: Path to save encrypted symmetric key
        :param pub_key_path: Path to save public RSA key
        :param private_key_path: Path to save private RSA key
        """
        try:
            # Symmetric key generation
            sym_key = os.urandom(sym_key_length // 8)

            # Asymmetric key generation
            private_key, public_key = AsymmetricCipher.generate_keys()

            # Encrypt and save symmetric key
            encrypted_key = AsymmetricCipher.encrypt(public_key, sym_key)
            FileManager.save_key(encrypted_key, enc_key_path)

            # Serialize keys
            SerializeKeys.serialize_public_key(public_key, pub_key_path)
            SerializeKeys.serialize_private_key(private_key, private_key_path)
        except Exception as e:
            raise Exception(f"Error at generating keys: {str(e)}")

    @staticmethod
    def encrypt_file(input_path, output_path, private_key_path, enc_key_path):
        """
        Encrypts file using hybrid cryptosystem

        :param input_path: Path to plaintext file
        :param output_path: Path to save encrypted file
        :param private_key_path: Path to RSA private key
        :param enc_key_path: Path to encrypted symmetric key
        """
        try:
            # Load and decrypt symmetric key
            private_key_data = FileManager.read_file(private_key_path)
            private_key = AsymmetricCipher.load_private_key(private_key_data)

            encrypted_key = FileManager.load_key(enc_key_path)
            sym_key = AsymmetricCipher.decrypt(private_key, encrypted_key)

            # Encrypt file
            data = FileManager.read_file(input_path)
            cipher = SymmetricCipher(sym_key)
            encrypted_data = cipher.encrypt(data)
            FileManager.write_file(output_path, encrypted_data)
        except Exception as e:
            raise Exception(f"Error at encrypting file: {str(e)}")

    @staticmethod
    def decrypt_file(input_path, output_path, private_key_path, enc_key_path):
        """
        Decrypts file using hybrid cryptosystem

        :param input_path: Path to encrypted file
        :param output_path: Path to save decrypted file
        :param private_key_path: Path to RSA private key
        :param enc_key_path: Path to encrypted symmetric key
        """
        try:
            # Load and decrypt symmetric key
            private_key_data = FileManager.read_file(private_key_path)
            private_key = AsymmetricCipher.load_private_key(private_key_data)

            encrypted_key = FileManager.load_key(enc_key_path)
            sym_key = AsymmetricCipher.decrypt(private_key, encrypted_key)

            # Decrypt file
            data = FileManager.read_file(input_path)
            cipher = SymmetricCipher(sym_key)
            decrypted_data = cipher.decrypt(data)
            FileManager.write_file(output_path, decrypted_data)
        except Exception as e:
            raise Exception(f"Error at decrypting file: {str(e)}")
