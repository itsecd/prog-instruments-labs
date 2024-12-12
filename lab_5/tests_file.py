import pytest
import os
import random

from algoritms.tripldes import SymmetricKey
from algoritms.rsa import AsymmetricKey


@pytest.fixture
def generate_sim_key():
    key_len = random.choice([64, 128, 192])
    key = os.urandom(key_len // 8)
    return key


@pytest.fixture
def symmetric():
    return SymmetricKey()


@pytest.fixture
def asymmetric():
    return AsymmetricKey()
