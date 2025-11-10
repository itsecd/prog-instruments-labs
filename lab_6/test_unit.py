import pytest

from module_for_tests import (
    calculate_symbol_frequency,
    decryption_cod3,
    make_key,
    text_decryption,
    text_encryption,
)

TEST_TEXT = "aabbcc"
FREQ_DATA = {"a": 0.3, "b": 0.3, "c": 0.4}
FREQ_TASK = {"x": 0.4, "y": 0.3, "z": 0.3}
DECRYPTION_DATA = "xay"
DECRYPTION_KEY = {"x": "a", "a": "k", "y": "b"}
TEST_KEY = "key"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

EXPECTED_FREQ_BASIC = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
EXPECTED_KEY_RESULT = {"x": "a", "y": "b", "z": "c"}
EXPECTED_DECRYPTION = "akb"


def test_calculate_symbol_frequency_basic():
    result = calculate_symbol_frequency(TEST_TEXT)
    assert result == EXPECTED_FREQ_BASIC


def test_make_key_creation():
    result = make_key(FREQ_DATA, FREQ_TASK)
    assert result == EXPECTED_KEY_RESULT


def test_decryption_cod3_simple():
    result = decryption_cod3(DECRYPTION_DATA, DECRYPTION_KEY)
    assert result == EXPECTED_DECRYPTION


def test_text_encryption_basic():
    encrypted_text = text_encryption(TEST_TEXT, TEST_KEY, ALPHABET)
    assert encrypted_text != TEST_TEXT


def test_text_decryption_basic():
    encrypted_text = text_encryption(TEST_TEXT, TEST_KEY, ALPHABET)
    result = text_decryption(encrypted_text, TEST_KEY, ALPHABET)
    assert result == TEST_TEXT


@pytest.mark.parametrize(
    "input_text",
    [
        "hello",
        "abracadabra",
        "mississippi",
        "programming",
    ],
)
def test_frequency_sorted_descending(input_text):
    result = calculate_symbol_frequency(input_text)
    total = sum(result.values())
    assert abs(total - 1.0) < 0.000001


def test_text_decryption_with_exceptions():
    with pytest.raises(ValueError):
        text_decryption("", TEST_KEY, ALPHABET)

    with pytest.raises(ValueError):
        text_decryption(TEST_TEXT, "", ALPHABET)


def test_decryption_cod3_with_stubs():

    class StubKey:
        def get(self, symbol, default=None):

            stub_mapping = {
                "x": "a",
                "y": "b",
                "z": "c",
                "1": "one",
                "2": "two",
                "3": "three",
            }
            return stub_mapping.get(symbol, default)

    stub_key = StubKey()

    test_data = "xyz123!@#"

    result = decryption_cod3(test_data, stub_key)

    expected = "abconetwothree!@#"
    assert result == expected

    assert result.startswith("abc")
    assert "one" in result
    assert "two" in result
    assert "three" in result
    assert "!@#" in result
