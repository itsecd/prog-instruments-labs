"""Characterization tests: freeze current behavior before refactoring."""
from pathlib import Path

import pytest

from checksum import calculate_checksum
from main import validate_csv

DATA = Path(__file__).resolve().parent.parent / "26.csv"
EXPECTED_CHECKSUM = "bd74879b1889a27178f3e49b3a52c4dc"


@pytest.fixture(scope="module")
def result():
    return validate_csv(DATA)


def test_row_count(result):
    assert result.row_count == 10_000


def test_invalid_rows_count(result):
    assert len(result.invalid_rows) == 1_000


def test_each_field_has_hundred_errors(result):
    for field, count in result.errors_by_field.items():
        assert count == 100, f"{field}: expected 100 invalid, got {count}"


def test_checksum_matches(result):
    checksum = calculate_checksum(result.invalid_rows.copy())
    assert checksum == EXPECTED_CHECKSUM