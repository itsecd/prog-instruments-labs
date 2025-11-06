import pytest

from asymmetric import Asymmetric
from cryptosistem import CryptoSistem
from symmetric import Symmetric


TEST_KEY = b"test_symmetric_key"
TEST_TEXT = b"Text for testing"
KEY_LEN = 256 # for cryptosistem
LEN_IV = 16


def test_generate_asymmetric_keys():

    """
    Test if generated keys is not empty
    :return: None
    """

    private_key, public_key = Asymmetric.generate_asymmetric_keys()

    assert private_key is not None
    assert public_key is not None


def test_encrypt_decrypt_symmetric_key():

    """
    Test if original key and encrypted/decrypted key is the same
    :return: None
    """

    private_key, public_key = Asymmetric.generate_asymmetric_keys()

    encrypted_key = Asymmetric.encrypt_symmetric_key(public_key, TEST_KEY)
    decrypted_key = Asymmetric.decrypt_symmetric_key(private_key, encrypted_key)

    assert decrypted_key == TEST_KEY
    assert encrypted_key != TEST_KEY


def test_generate_symmetric_key():

    """
    Test generating different length keys
    :return:
    """

    key_128 = Symmetric.generate_symmetric_key(128)
    key_256 = Symmetric.generate_symmetric_key(256)

    assert len(key_128) == 16
    assert len(key_256) == 32
    assert key_128 != key_256


def test_encrypt_decrypt_text():

    """
    Test of text encryption and decryption
    :return: None
    """

    key = Symmetric.generate_symmetric_key(KEY_LEN)
    iv = Symmetric.iv()

    encrypted_text = Symmetric.encrypt_text(TEST_TEXT, key, iv)
    decrypted_text = Symmetric.decrypt_text(encrypted_text, key, iv)

    assert decrypted_text == TEST_TEXT
    assert encrypted_text != TEST_TEXT


def test_cryptosistem_initialization():

    """
    Test if cryptosistem initialized/initialized correctly
    :return: None
    """

    crypto = CryptoSistem(KEY_LEN)

    assert crypto._key_len == KEY_LEN
    assert crypto._iv is not None
    assert len(crypto._iv) == LEN_IV


def test_cryptosistem_generate_hybrid_keys():

    """
    Test of key generating in cryptosistem
    :return: None
    """

    crypto = CryptoSistem(KEY_LEN)

    c_symmetric_key, public_key, private_key = crypto.generate_hybrid_keys()

    assert c_symmetric_key is not None
    assert public_key is not None
    assert private_key is not None
    assert len(c_symmetric_key) == KEY_LEN


@pytest.mark.parametrize("texts", [
    b"Short",
    b"Medium length text",
    b"Very long text " * 10,
    b""
])
def test_encrypt_decrypt_different_texts(texts):

    """
    Test encryption/decryption text of different length
    :param text_input: Texts with different lengths
    :return: None
    """

    key = Symmetric.generate_symmetric_key(KEY_LEN)
    iv = Symmetric.iv()

    encrypted = Symmetric.encrypt_text(texts, key, iv)
    decrypted = Symmetric.decrypt_text(encrypted, key, iv)

    assert decrypted == texts


def test_generate_hybrid_keys(monkeypatch):

    """
    Test of generate_hybrid_keys function working
    :return: None
    """

    def mock_gen_symmetric(len_key):
        return b'mock_symmetric_key'

    def mock_gen_asymmetric():
        return ('mock_private', 'mock_public')

    def mock_encrypt_symmetric(public_key, symmetric_key):
        return b'mock_encrypted_key'

    monkeypatch.setattr('symmetric.Symmetric.generate_symmetric_key', mock_gen_symmetric)
    monkeypatch.setattr('asymmetric.Asymmetric.generate_asymmetric_keys', mock_gen_asymmetric)
    monkeypatch.setattr('asymmetric.Asymmetric.encrypt_symmetric_key', mock_encrypt_symmetric)

    crypto = CryptoSistem(KEY_LEN)
    result = crypto.generate_hybrid_keys()

    assert result == (b'mock_encrypted_key', 'mock_public', 'mock_private')
