import os
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from asymmetrical import Asymmetrical
from symmetrical import Symmetrical
from filehandler import FileHandler
from hybrid import Hybrid
import main


# 1. Тест генерации RSA-ключей
def test_generate_asymmetrical_keys():
    private_key, public_key = Asymmetrical.generate_asymmetrical_keys()
    assert private_key is not None
    assert public_key is not None
    assert private_key.key_size == 2048

