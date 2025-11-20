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
    """Тесты для реализации алгоритма Луна.

    Этот класс содержит тесты для проверки номеров кредитных карт
    с использованием алгоритма Луна и вычисления контрольных цифр.
    """

    def test_luhn_check_valid_card(self):
        """Тест проверки заведомо валидного номера кредитной карты.

        Проверяет, что алгоритм Луна корректно идентифицирует
        валидный номер кредитной карты из реального примера.
        """
        # Используем заведомо валидный номер карты
        assert luhn_algorithm_check("4532015112830366") == True

    def test_luhn_check_invalid_card(self):
        """Тест проверки невалидного номера кредитной карты.

        Проверяет, что алгоритм Луна корректно идентифицирует
        невалидный номер кредитной карты (изменен один символ).
        """
        assert luhn_algorithm_check("4532015112830367") == False

    @pytest.mark.parametrize("number,expected", [
        ("4532015112830366", True),  # Валидная Visa карта
        ("6011514433546201", True),  # Валидная Discover карта
        ("1234567812345670", True),  # Валидный тестовый номер
        ("1234567812345678", False),  # Невалидный номер
        ("1111111111111111", False),  # Невалидный номер
    ])
    def test_luhn_algorithm_check_parametrized(self, number, expected):
        """Параметризованный тест проверки алгоритма Луна с различными номерами.

        Args:
            number (str): Номер карты для проверки
            expected (bool): Ожидаемый результат проверки
        """
        assert luhn_algorithm_check(number) == expected

    def test_luhn_algorithm_compute(self):
        """Тест вычисления контрольной цифры по алгоритму Луна.

        Проверяет, что функция вычисления контрольной цифры
        корректно добавляет контрольную цифру к базовому номеру
        и полученный номер проходит проверку валидности.
        """
        result = luhn_algorithm_compute("123456781234567")
        assert result == "1234567812345670"
        assert luhn_algorithm_check(result) == True


class TestCardSearcher:
    """Тесты для класса поиска номеров кредитных карт.

    Этот класс содержит тесты для функциональности генерации
    и поиска номеров кредитных карт по их хешу.
    """

    def test_generate_card_numbers(self):
        """Тест генерации номеров кредитных карт для заданного BIN.

        Проверяет, что функция генерации создает корректное количество
        номеров карт с правильным форматом и последовательностью.
        """
        searcher = CardSearcher()

        with patch('hash_card.LAST_4_DIGITS', '2301'):
            numbers = searcher.generate_card_numbers("547905")

            assert len(numbers) == 1000000
            assert numbers[0] == "5479050000002301"
            assert numbers[999999] == "5479059999992301"

    @patch('hash_card.multiprocessing.Pool')
    @patch('hash_card.BINS', ['547905'])
    def test_search_card_number_with_mocks(self, mock_pool):
        """Продвинутый тест поиска карты с использованием мок-объектов.

        Тестирует процесс поиска номера карты с подменой многопроцессорного пула
        и проверяет корректность возвращаемых результатов.

        Args:
            mock_pool: Мок-объект для multiprocessing.Pool
        """
        # Настраиваем мок пула процессов
        mock_pool_instance = Mock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool_instance.imap_unordered.return_value = ["5479054156572301"]

        searcher = CardSearcher()
        # Мокаем внутренние атрибуты для изоляции теста
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
    """Тесты для бенчмарка производительности поиска карт.

    Этот класс содержит тесты для измерения и анализа
    производительности поиска номеров карт при разном количестве процессов.
    """

    def test_initialization(self):
        """Тест инициализации бенчмарка производительности.

        Проверяет корректность установки начальных параметров
        при создании экземпляра PerformanceBenchmark.
        """
        test_bins = ["547905", "546925"]
        benchmark = PerformanceBenchmark(test_bins)

        assert benchmark.bins == test_bins
        assert benchmark.results == []

    @patch('time_test.CardSearcher')
    def test_run_tests_with_mocks(self, mock_searcher_class):
        """Продвинутый тест выполнения тестов производительности с моками.

        Тестирует процесс запуска серии тестов производительности
        с подменой CardSearcher и проверяет сбор статистики.

        Args:
            mock_searcher_class: Мок-класс для CardSearcher
        """
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