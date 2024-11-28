import pytest

from hybrid import HybridCryptography
from asymmetric import AsymmetricCryptography
from symmetric import SymmetricCryptography
from functions import ReadWriteParseFunctions
from constants import PATHS

from cryptography.hazmat.primitives.asymmetric import rsa


@pytest.fixture
def asym_crypto():
    return AsymmetricCryptography(private_key_path="private.pem", public_key_path="public.pem")


def test_asym_generate_keys(asym_crypto):
    private_key, public_key = asym_crypto.generate_key(size=2048)
    assert isinstance(private_key, rsa.RSAPrivateKey)
    assert isinstance(public_key, rsa.RSAPublicKey)


def test_asym_encrypt_decrypt(asym_crypto):
    msg = b'Testing Asymmetric Cryptography'
    private_key, public_key = asym_crypto.generate_key(size=2048)
    enc_msg = asym_crypto.encrypt(msg, public_key)
    assert enc_msg != msg
    dec_msg = asym_crypto.decrypt(msg, private_key)
    assert dec_msg == msg


@pytest.fixture
def sym_crypto():
    return SymmetricCryptography(symmetric_key_path="symmetric.txt", private_key_path="private.pem", public_key_path="public.pem")


@pytest.mark.parametrize("key_size", [16, 24, 32])
def test_sym_generate_key(sym_crypto, key_size):
    key = sym_crypto.generate_key(size=key_size)
    assert len(key) == key_size
    assert isinstance(key, bytes)


@pytest.mark.parametrize("key_size", [16, 24, 32])
def test_sym_encrypt_decrypt(asym_crypto, key_size):
    msg = b'Testing Symmetric Cryptography'
    key = asym_crypto.generate_key(size=key_size)
    enc_msg = asym_crypto.encrypt(msg, key)
    assert enc_msg != msg
    dec_msg = asym_crypto.decrypt(msg, key)
    assert dec_msg == msg

    
