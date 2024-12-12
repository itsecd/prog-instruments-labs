from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from files_funct import FilesFunct


class AsymmetricKey:
    def __init__(self):
        """
        Constructor for the AsymmetricKey class.
        """
        pass

    def generate_keys(self) -> tuple[rsa.RSAPublicKey, rsa.RSAPrivateKey]:
        """
        Generates a pair of asymmetric keys: private and public.

        Return:
                tuple: public key and private key
        """
        keys = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_key = keys
        public_key = keys.public_key()
        return (public_key, private_key)

    def encrypt_symmetric_key(
        self, symmetric_key_path: str, public_key_path: str, encrypt_sym_path: str
    ) -> None:
        """
        Encrypts a symmetric key using an RSA public key.
        """
        public_key = FilesFunct.deserialization_rsa_public_key(public_key_path)
        sym_key = FilesFunct.deserial_sym_key(symmetric_key_path)
        c_text = public_key.encrypt(
            sym_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        FilesFunct.write_bytes_to_file(encrypt_sym_path, c_text)

    def decrypt_symmetric_key(
        self, symmetric_en_key_path: str, private_key_path: str, decrypt_key_path: str
    ) -> None:
        """
        Decrypts a symmetric key using an RSA private key.
        """
        private_key = FilesFunct.deserialization_rsa_private_key(
            private_key_path)
        sym_key = FilesFunct.deserial_sym_key(symmetric_en_key_path)
        dc_text = private_key.decrypt(
            sym_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        FilesFunct.write_bytes_to_file(decrypt_key_path, dc_text)
