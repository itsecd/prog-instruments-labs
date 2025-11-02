import pytest
from asymmetric import Asymmetric
from symmetric import Symmetric
from cryptosistem import CryptoSistem


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
    test_key = b"test_symmetric_key_32b"

    encrypted_key = Asymmetric.encrypt_symmetric_key(public_key, test_key)
    decrypted_key = Asymmetric.decrypt_symmetric_key(private_key, encrypted_key)

    assert decrypted_key == test_key
    assert encrypted_key != test_key