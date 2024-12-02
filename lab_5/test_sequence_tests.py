import pytest
from unittest.mock import patch, mock_open
from works_files import read_json, write_files
from test import frequency_test, same_bits_test, longest_sequence_in_block_test

# Sample data for testing
sample_json_data = {
    "cpp": "00011111100111011111111001011000001000001101001101010001101001100101101111100000110011110110100000110101010101100110010000100100",
    "java": "10110100000011101011100110101111110000000100111110011000100111001001011101111111001011001011000100101101101010110111010001101110"
}


# Тесты
def test_read_json():
    with patch("builtins.open", mock_open(read_data='{"cpp": "000111111", "java": "101101000"}')):
        data = read_json("dummy_path.json")
        assert data["cpp"] == "000111111"
        assert data["java"] == "101101000"


def test_write_files():
    mock_file = mock_open()
    with patch("builtins.open", mock_file):
        write_files("dummy_path.txt", "Test data")
        mock_file().write.assert_called_once_with("Test data")


def test_read_json_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError):
        data = read_json("non_existing_file.json")
        assert data is None  # Проверяем, что возвращается None


@patch('test.read_json')
@patch('test.write_files')
def test_frequency_test(mock_write, mock_read):
    mock_read.return_value = sample_json_data
    frequency_test("dummy_path.json", "dummy_output.txt", "java")
    mock_write.assert_called_once()


@patch('test.read_json')
@patch('test.write_files')
def test_same_bits_test(mock_write, mock_read):
    mock_read.return_value = sample_json_data
    same_bits_test("dummy_path.json", "dummy_output.txt", "cpp")
    mock_write.assert_called_once()


@patch('test.read_json')
@patch('test.write_files')
def test_longest_sequence_in_block_test(mock_write, mock_read):
    mock_read.return_value = sample_json_data
    longest_sequence_in_block_test("dummy_path.json", "dummy_output.txt", "java")
    mock_write.assert_called_once()


@pytest.mark.parametrize("seq_key, expected_value", [
    ("java", "Частотный побитовый тест java : 0.3767591178115821\n"),
    ("cpp", "Частотный побитовый тест cpp : 0.8596837951986662\n")
])
@patch('test.read_json')
@patch('test.write_files')
def test_parameterized_frequency_test(mock_write, mock_read, seq_key, expected_value):
    mock_read.return_value = sample_json_data
    frequency_test("dummy_path.json", "dummy_output.txt", seq_key)
    mock_write.assert_called_once_with("dummy_output.txt", expected_value)
