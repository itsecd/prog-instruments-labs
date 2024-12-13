from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key


class Asymmetric:
    """
    Class for implementing asymmetric algorithms
    """
    def __init__(self) -> None:
        self.private_key = None
        self.public_key = None

    def generate_key(self) -> None:
        """
        Generate a new RSA keys.
        """
        keys = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
        )
        self.private_key = keys
        self.public_key = keys.public_key()

    def serialize_public_key(self, public_path: str) -> None:
        """
        Serialize the public key and save it to a file.
        """
        with open(public_path, 'wb') as public_out:
            public_out.write(self.public_key.public_bytes(encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo))
            
    def serialize_private_key(self, private_path: str) -> None:
        """
        Serialize the private key and save it to a file.
        """
        with open(private_path, 'wb') as private_out:
            private_out.write(self.private_key.private_bytes(encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()))
    
    def deserialize_public_key(self, public_path: str) -> None:
        """
        Deserialize the public key from a file.
        """
        with open(public_path, 'rb') as pem_in:
            public_bytes = pem_in.read()
        self.public_key = load_pem_public_key(public_bytes)
    
    def deserialize_private_key(self, private_path: str) -> None:
        """
        Deserialize the private key from a file.
        """
        with open(private_path, 'rb') as pem_in:
            private_bytes = pem_in.read()
        self.private_key = load_pem_private_key(private_bytes,password=None,)

    def encrypt_key(self, key: bytes) -> bytes:
        """
        Encrypt a symmetric key using RSA encryption.
        """
        c_key = self.public_key.encrypt(
            key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return c_key

    def decrypt_key(self, key: bytes) -> bytes:
        """
        Decrypt an encrypted symmetric key using RSA decryption.
        """
        dc_text = self.private_key.decrypt(
            key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return dc_text