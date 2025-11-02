import pytest
from asymmetric import Asymmetric
from symmetric import Symmetric
from cryptosistem import CryptoSistem


TEST_KEY = b"test_symmetric_key"
TEST_TEXT = b"Hello, this is a secret message!"


def test_generate_asymmetric_keys():

    '''
    Test if generated keys is not empty
    :return: None
    '''

    private_key, public_key = Asymmetric.generate_asymmetric_keys()

    assert private_key is not None
    assert public_key is not None

def test_encrypt_decrypt_symmetric_key():

    '''
    Test if original key and encrypted/decrypted key is the same
    :return: None
    '''

    private_key, public_key = Asymmetric.generate_asymmetric_keys()

    encrypted_key = Asymmetric.encrypt_symmetric_key(public_key, TEST_KEY)
    decrypted_key = Asymmetric.decrypt_symmetric_key(private_key, encrypted_key)

    assert decrypted_key == TEST_KEY
    assert encrypted_key != TEST_KEY

def test_generate_symmetric_key():

    '''
    Test generating different length keys
    :return:
    '''

    key_128 = Symmetric.generate_symmetric_key(128)
    key_256 = Symmetric.generate_symmetric_key(256)

    assert len(key_128) == 16
    assert len(key_256) == 32
    assert key_128 != key_256

def test_encrypt_decrypt_text():

    '''
    Test of text encryption and decryption
    :return: None
    '''

    key = Symmetric.generate_symmetric_key(256)
    iv = Symmetric.iv()

    encrypted_text = Symmetric.encrypt_text(TEST_TEXT, key, iv)
    decrypted_text = Symmetric.decrypt_text(encrypted_text, key, iv)

    assert decrypted_text == TEST_TEXT
    assert encrypted_text != TEST_TEXT