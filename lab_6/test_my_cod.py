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
        # Используем заведомо валидный номер карты
        assert luhn_algorithm_check("4532015112830366") == True

    def test_luhn_check_invalid_card(self):
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


class TestCardSearcher:

    def test_generate_card_numbers(self):
        searcher = CardSearcher()

        with patch('hash_card.LAST_4_DIGITS', '2301'):
            numbers = searcher.generate_card_numbers("547905")

            assert len(numbers) == 1000000
            assert numbers[0] == "5479050000002301"
            assert numbers[999999] == "5479059999992301"

    @patch('hash_card.multiprocessing.Pool')
    @patch('hash_card.BINS', ['547905'])
    def test_search_card_number_with_mocks(self, mock_pool):
        # Настраиваем мок пула процессов
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.imap_unordered.return_value = ["5479054156572301"]

        searcher = CardSearcher()
        searcher.found_event = Mock()
        searcher.found_event.is_set.return_value = False
        searcher.result_queue = Mock()
        searcher.result_queue.empty.return_value = False
        searcher.result_queue.get.return_value = "5479054156572301"

        result, duration = searcher.search_card_number(num_processes=2)

        assert result == "5479054156572301"
        assert isinstance(duration, float)
        mock_pool.assert_called_once()


class TestPerformanceBenchmark:

    def test_initialization(self):

        test_bins = ["547905", "546925"]
        benchmark = PerformanceBenchmark(test_bins)

        assert benchmark.bins == test_bins
        assert benchmark.results == []

    @patch('time_test.CardSearcher')
    def test_run_tests_with_mocks(self, mock_searcher_class):

        # Настраиваем мок CardSearcher
        mock_searcher = Mock()
        mock_searcher.search_card_number.return_value = ("5479054156572301", 5.0)
        mock_searcher_class.return_value = mock_searcher

        benchmark = PerformanceBenchmark(["547905"])
        benchmark.run_tests(3)

        assert len(benchmark.results) == 3
        assert benchmark.results == [(1, 5.0), (2, 5.0), (3, 5.0)]
        assert mock_searcher.search_card_number.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
