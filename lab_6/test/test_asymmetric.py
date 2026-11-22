import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from crypto.asymmetric import AsymmetricCrypto


@pytest.fixture
def asym(config):
    return AsymmetricCrypto(config)


def test_rsa_roundtrip(asym):
    private, public = asym.generate_rsa_keypair()
    assert isinstance(private, rsa.RSAPrivateKey)
    assert isinstance(public, rsa.RSAPublicKey)

    data = b"top secret"
    encrypted = asym.encrypt_with_public_key(data, public)
    assert asym.decrypt_with_private_key(encrypted, private) == data


def test_rsa_save_load_pair(asym):
    private, public = asym.generate_rsa_keypair()
    pub_pem = asym.save_public_key_to_bytes(public)
    priv_pem = asym.save_private_key_to_bytes(private)

    assert b"BEGIN PUBLIC KEY" in pub_pem
    assert b"BEGIN PRIVATE KEY" in priv_pem

    pub2 = asym.load_public_key(pub_pem)
    priv2 = asym.load_private_key(priv_pem)

    msg = b"serialized roundtrip"
    enc = asym.encrypt_with_public_key(msg, pub2)
    assert asym.decrypt_with_private_key(enc, priv2) == msg


@pytest.mark.parametrize("bad_config", [
    {},
    {"crypto": {}},
    {"crypto": {"rsa": None}},
])
def test_rsa_bad_config(bad_config):
    """Проверяет, что при неполной конфигурации выбрасывается исключение."""
    with pytest.raises((KeyError, TypeError)):
        AsymmetricCrypto(bad_config)