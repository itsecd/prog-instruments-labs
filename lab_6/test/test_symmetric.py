import pytest
from crypto.symmetric import SEEDCipher, ChaCha20Cipher


def test_seed_roundtrip(config):
    cipher = SEEDCipher(config)
    key = cipher.generate_key()
    plaintext = b"Hello, SEED world!"
    encrypted = cipher.encrypt(plaintext, key)
    assert len(encrypted) > 16
    assert cipher.decrypt(encrypted, key) == plaintext


def test_seed_invalid_config():
    with pytest.raises(ValueError):
        SEEDCipher({"crypto": {"seed": [32, 16]}})


def test_chacha_roundtrip(config):
    cipher = ChaCha20Cipher(config)
    key = cipher.generate_key()
    plaintext = "Привет, ChaCha20!".encode("utf-8")
    encrypted = cipher.encrypt(plaintext, key)
    assert len(encrypted) > 12
    assert cipher.decrypt(encrypted, key) == plaintext


def test_chacha_wrong_key_raises(config):
    cipher = ChaCha20Cipher(config)
    key = cipher.generate_key()
    wrong_key = cipher.generate_key()
    encrypted = cipher.encrypt(b"secret message", key)
    with pytest.raises(Exception):
        cipher.decrypt(encrypted, wrong_key)


@pytest.mark.parametrize(
    "cipher_cls, key_len",
    [
        (SEEDCipher, 16),
        (ChaCha20Cipher, 32),
    ],
)
def test_generate_key_length(config, cipher_cls, key_len):
    cipher = cipher_cls(config)
    key = cipher.generate_key()
    assert isinstance(key, bytes)
    assert len(key) == key_len
    assert cipher.get_key_size() == key_len