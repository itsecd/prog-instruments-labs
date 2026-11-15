from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
import os

def generate_sym_key(size)->bytes:
    """Генерация ключа симметричного алгоритма"""
    try:
        if size<32 or size>448 or size%8!=0:
            raise ValueError("Size should be in range from 32 to 442 in increments of 8 inclusive")
        return os.urandom(size//8)
    except ValueError as e:
        print(f"Error:{e}")
        raise

def generate_asy_key()->tuple[RSAPrivateKey, RSAPublicKey]:
    """Генерация открытых и закрытых ключей"""
    keys = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
    )
    private_key = keys
    public_key = keys.public_key()
    return private_key,public_key