import pytest
import os

from cryptography.hazmat.primitives.asymmetric import rsa

from files import FilesHelper
from symmetric import Symmetric
from asymmetric import Asymmetric


@pytest.fixture
def symmetric_key_():
    key = os.urandom(16)
    return key


@pytest.fixture
def asymmetric_keys():
    private_key = rsa.generate_private_key(
        public_exponent = 65537,
        key_size = 2048
    )
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.mark.parametrize("path", ["setings.json", "some.json"])
def test_get_json(path):
    with pytest.raises(Exception):
         FilesHelper.get_json(path)
