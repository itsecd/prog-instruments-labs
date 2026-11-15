"""Интеграционный тест полного прогона на реальном CSV."""
from checksum import calculate_checksum
from src.config import DATA_PATH, VARIANT
from main import process_csv


EXPECTED_INVALID_COUNT = 1000
EXPECTED_CHECKSUM = "ceea41d057c2bfc5682c8886e74dfc24"


def test_process_csv_real_file_count():
    """На реальном файле ровно 1000 невалидных строк."""
    invalid = process_csv(DATA_PATH)
    assert len(invalid) == EXPECTED_INVALID_COUNT


def test_process_csv_real_file_checksum():
    """Контрольная сумма совпадает с эталонной."""
    invalid = process_csv(DATA_PATH)
    assert calculate_checksum(invalid) == EXPECTED_CHECKSUM


def test_process_csv_returns_ints():
    """Номера строк — int, 0-based (см. докстринг calculate_checksum)."""
    invalid = process_csv(DATA_PATH)
    assert all(isinstance(n, int) for n in invalid)
    assert all(0 <= n < 10000 for n in invalid)


def test_variant_matches_config():
    """Номер варианта совпадает с тем, что в config.py."""
    assert VARIANT == 36
