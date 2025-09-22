from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

class Asymmetrical:
    @staticmethod
    def encrypt_symmetric_key(public_key, symmetric_key):
        """
        A function for encrypting a symmetric key using an RSA public key
        :param public_key: RSA public key
        :param symmetric_key: The symmetric key that needs to be encrypted
        :param path_to_save: Path to save encrypted symmetric key
        """
        print(f"Шифрование симметричного ключа с помощью открытого ключа...")
        try:
            encrypted_symmetric_key = public_key.encrypt(symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(
                        algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            print(f"||Симметричный ключ успешно зашифрован||")
            return encrypted_symmetric_key
        except Exception as e:
            print(f"Error: Произошла ошибка при шифровании ключа {e}")
            exit(1)

    @staticmethod
    def decrypt_symmetric_key(private_key, encrypted_symmetric_key):
        """
        A function for decrypting a symmetric key using a private RSA key
        :param private_key: Private key
        :param encrypted_symmetric_key: Encrypted symmetric key
        :return: Decrypted symmetric key
        """
        print("Расшифровка симметричного ключа в процессе...")
        try:
            decrypted_symmetric_key = private_key.decrypt(
                encrypted_symmetric_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            print("||Симметричный ключ успешно расшифрован!||")
            return decrypted_symmetric_key
        except Exception as e:
            print(f"Error: Произошла ошибка при расшифровке симметричного ключа {e}")
            exit(1)