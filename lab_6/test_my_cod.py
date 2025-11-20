import pytest
import json
import hashlib
from unittest.mock import Mock, patch
import tempfile
import os

from hash_card import CardSearcher
from lun import luhn_algorithm_check, luhn_algorithm_compute
from time_test import PerformanceBenchmark


class TestLuhnAlgorithm:

    def test_luhn_check_valid_card(self):
        assert luhn_algorithm_check("4532015112830366") == True

    def test_luhn_check_invalid_card(self):
        assert luhn_algorithm_check("4532015112830367") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])