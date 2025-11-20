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
    """Тесты для алгоритма Луна"""

    def test_luhn_check_valid_card(self):
        """Тест проверки валидного номера карты"""
        # Используем заведомо валидный номер карты
        assert luhn_algorithm_check("4532015112830366") == True

    def test_luhn_check_invalid_card(self):
        """Тест проверки невалидного номера карты"""
        assert luhn_algorithm_check("4532015112830367") == False

    @pytest.mark.parametrize("number,expected", [
        ("4532015112830366", True),
        ("6011514433546201", True),
        ("1234567812345670", True),
        ("1234567812345678", False),
        ("1111111111111111", False),
    ])
    def test_luhn_algorithm_check_parametrized(self, number, expected):
        assert luhn_algorithm_check(number) == expected

    def test_luhn_algorithm_compute(self):
        result = luhn_algorithm_compute("123456781234567")
        assert result == "1234567812345670"
        assert luhn_algorithm_check(result) == True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])