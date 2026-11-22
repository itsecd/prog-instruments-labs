import pytest
from crypto.hybrid import HybridCrypto


@pytest.fixture
def hybrid(config):
    return HybridCrypto(config)


@pytest.fixture
def rsa_pair(hybrid):
    return hybrid.asymmetric.generate_rsa_keypair()


@pytest.mark.parametrize("algorithm", ["SEED", "ChaCha20"])
def test_hybrid_roundtrip(hybrid, rsa_pair, algorithm):
    private, public = rsa_pair
    plaintext = b"hybrid secret payload " * 3
    enc_data, enc_key = hybrid.encrypt_hybrid(plaintext, algorithm, public)
    assert hybrid.decrypt_hybrid(enc_data, enc_key, algorithm, private) == plaintext


def test_hybrid_wrong_private_key(hybrid, rsa_pair):
    """ Проверяет, что расшифрование не сработает при использовании неправильного приватного ключа."""
    _, public = rsa_pair
    other_private, _ = hybrid.asymmetric.generate_rsa_keypair()
    enc_data, enc_key = hybrid.encrypt_hybrid(b"msg", "ChaCha20", public)
    with pytest.raises(Exception):
        hybrid.decrypt_hybrid(enc_data, enc_key, "ChaCha20", other_private)