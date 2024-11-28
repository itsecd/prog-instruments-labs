import pytest
import os

from unittest.mock import patch, MagicMock
from cryptography.hazmat.primitives.asymmetric import rsa

from hybrid import HybridCryptography
from asymmetric import AsymmetricCryptography
from symmetric import SymmetricCryptography
from functions import ReadWriteParseFunctions
from constants import PATHS


@pytest.fixture
def asym_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


def test_serialize_asym_keys(rsa_keys):
    private_key, public_key = rsa_keys
    ReadWriteParseFunctions.serialize_private_key(path='private.pem', private_key=private_key)
    ReadWriteParseFunctions.serialize_public_key(path='public.pem', public_key=public_key)
    written_private_key = ReadWriteParseFunctions.deserialize_private_key(path='private.pem')
    written_public_key = ReadWriteParseFunctions.deserialize_public_key(path='public.pem')
    assert written_private_key == private_key
    assert written_public_key == public_key


@pytest.fixture
def sym_key():
    key = os.urandom(size=32)
    return key


def test_serialize_sym_key(sym_key):
    ReadWriteParseFunctions.write_bytes(path='sym_key.txt', data=sym_key)
    written_sym_key = ReadWriteParseFunctions.read_bytes(path='sym_key.txt')
    assert written_sym_key == sym_key


@pytest.fixture
def asym_crypto():
    return AsymmetricCryptography(private_key_path='private.pem', public_key_path='public.pem')


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
    return SymmetricCryptography(symmetric_key_path='symmetric.txt', private_key_path='private.pem', public_key_path='public.pem')


@pytest.mark.parametrize('key_size', [16, 24, 32])
def test_sym_generate_key(sym_crypto, key_size):
    key = sym_crypto.generate_key(size=key_size)
    assert len(key) == key_size
    assert isinstance(key, bytes)


@pytest.mark.parametrize('key_size', [16, 24, 32])
def test_sym_encrypt_decrypt(asym_crypto, key_size):
    msg = b'Testing Symmetric Cryptography'
    key = asym_crypto.generate_key(size=key_size)
    enc_msg = asym_crypto.encrypt(msg, key)
    assert enc_msg != msg
    dec_msg = asym_crypto.decrypt(msg, key)
    assert dec_msg == msg


@pytest.fixture
def hybrid_crypto():
    return HybridCryptography(
        symmetric_key_path='symmetric.key',
        private_key_path='private.pem',
        public_key_path='public.pem')


def test_hybrid_generate_keys(hybrid_crypto):
    symmetric_key = MagicMock()
    private_key = MagicMock()
    public_key = MagicMock()
    with patch('symmetric.SymmetricCryptography.generate_key', return_value=symmetric_key), \
         patch('asymmetric.AsymmetricCryptography.generate_key', return_value=(private_key, public_key)), \
         patch('functions.ReadWriteParseFunctions.serialize_private_key') as serialize_private_key, \
         patch('functions.ReadWriteParseFunctions.serialize_public_key') as serialize_public_key, \
         patch('functions.ReadWriteParseFunctions.write_bytes') as write_bytes, \
         patch.object(public_key, 'encryp', return_value=b'encrypted_symmetric_key'):
        hybrid_crypto.generate_keys(size=32)
        serialize_private_key.assert_called_once_with(hybrid_crypto.asymmetric.private_key_path, private_key)
        serialize_public_key.assert_called_once_with(hybrid_crypto.asymmetric.public_key_path, public_key)
        write_bytes.assert_called_once_with(hybrid_crypto.symmetric.key_path, b'encrypted_symmetric_key')


def test_hybrid_encrypt_decrypt(hybrid_crypto):
    msg = 'Testing Hybrid Cryptography'
    symmetric_key = b'symmetric_key'
    encrypted_symmetric_key = b'encrypted_symmetric_key'
    enc_msg = b'enc_msg'
    dec_msg = b'Testing Hybrid Cryptography'
    with patch('functions.ReadWriteParseFunctions.read_txt', return_value=msg), \
         patch('functions.ReadWriteParseFunctions.deserialize_private_key', return_value=MagicMock()), \
         patch('functions.ReadWriteParseFunctions.read_bytes', side_effect=[encrypted_symmetric_key, enc_msg]), \
         patch('asymmetric.AsymmetricCryptography.decrypt', return_value=symmetric_key), \
         patch('symmetric.SymmetricCryptography.encrypt', return_value=enc_msg), \
         patch('symmetric.SymmetricCryptography.decrypt', return_value=dec_msg), \
         patch('functions.ReadWriteParseFunctions.write_bytes') as write_bytes:
        hybrid_crypto.encrypt(text_path='msg.txt', encrypted_msg_path='enc_msg.txt')
        ReadWriteParseFunctions.read_txt.assert_called_once_with('msg.txt')
        ReadWriteParseFunctions.deserialize_private_key.assert_called_once_with(
            hybrid_crypto.asymmetric.private_key_path)
        ReadWriteParseFunctions.read_bytes.assert_any_call(hybrid_crypto.symmetric.key_path)
        AsymmetricCryptography.decrypt.assert_called_once_with(
            encrypted_symmetric_key,
            ReadWriteParseFunctions.deserialize_private_key.return_value)
        SymmetricCryptography.encrypt.assert_called_once_with(bytes(msg, 'UTF-8'), symmetric_key)
        write_bytes.assert_any_call('enc_msg.txt', enc_msg)
        hybrid_crypto.decrypt(text_path='enc_msg.txt', decrypted_text_path='dec_msg.txt')
        ReadWriteParseFunctions.read_bytes.assert_any_call('enc_msg.txt')
        ReadWriteParseFunctions.deserialize_private_key.assert_called_with(
            hybrid_crypto.asymmetric.private_key_path)
        SymmetricCryptography.decrypt.assert_called_once_with(enc_msg, symmetric_key)
        write_bytes.assert_any_call('enc_msg.txt', enc_msg)
 