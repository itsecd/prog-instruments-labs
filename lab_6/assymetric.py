from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from symmetric import symmetric_key_generation

def asymmetric_keys_generation():
    """
    Generate asymmetric keys

    Returns:
        tuple: (private_key, public_key) where private_key is RSA private key and public_key is corresponding RSA public key
    """
    keys = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    private_key = keys
    public_key = keys.public_key()

    return private_key, public_key


def secret_key_writter(secret_pem: str, private_key: bytes):
    """
    Write secret key in pem file

    Args:
        secret_pem: Path to output private key file
        private_key: Private key to write
    """
    with open(secret_pem, 'wb') as private_out:
        private_out.write(private_key.private_bytes(encoding=serialization.Encoding.PEM,
              format=serialization.PrivateFormat.TraditionalOpenSSL,
              encryption_algorithm=serialization.NoEncryption()))


def public_key_writter(public_pem: str, public_key: bytes):
    """
    Write public key in pem file

    Args:
        public_pem: Path to output public key file
        public_key: Public key to write
    """
    with open(public_pem, 'wb') as public_out:
        public_out.write(public_key.public_bytes(encoding=serialization.Encoding.PEM,
             format=serialization.PublicFormat.SubjectPublicKeyInfo))


def asymmetric_keys_writter(secret_pem: str, public_pem: str, asymmetric_keys: tuple):
    """
    Write asymmetric keys

    Args:
        secret_pem: Path to output private key file
        public_pem: Path to output public key file
        asymmetric_keys: Tuple containing (private_key, public_key)
    """
    secret_key_writter(secret_pem, asymmetric_keys[0])
    public_key_writter(public_pem, asymmetric_keys[1])


def get_secret_key(filename: str):
    """
    Read asymmetric key from file

    Args:
        filename: Path to the pem private key file
    
    Returns:
        Private key
    """
    try:
        with open(filename, 'rb') as file:
            private_bytes = file.read()
        return load_pem_private_key(private_bytes,password=None,)
    except Exception as e:
        print(f"Error: {e}")
        raise e


def key_encription(symmetric_key: bytes, public_key: bytes):
    """
    Symmetric key encryption

    Args:
        symmetric_key: Symmetric key bytes
        public_key: Public key for encryption
    
    Returns:
        bytes: Encrypted symmetric key
    """
    try:
        return public_key.encrypt(symmetric_key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
    except ValueError:
        symmetric_key = symmetric_key_generation()
        return public_key.encrypt(symmetric_key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None))


def decrypt_key(key: bytes, secret_key: bytes) -> bytes:
    """
    Decryption of symmetric key

    Args:
        key: Encrypted symmetric key bytes
        secret_key: Private key
    
    Returns:
        bytes: Decrypted symmetric key
    """
    try:
        return secret_key.decrypt(key,padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
    except ValueError:
        return key
    except Exception as e:
        print(f"Error: {e}")
        raise e