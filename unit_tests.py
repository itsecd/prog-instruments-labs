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