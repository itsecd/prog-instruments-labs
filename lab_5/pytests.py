import os
import pytest
from utility.functions import read_json, write_text
from utility.tests import frequency_bitwise_test, consecutive_bits_test, longest_sequence_test


@pytest.mark.parametrize(
    "test_function, sequence, expected_range",
    [
        (frequency_bitwise_test, "1110001010101001", (0, 1)),
        (consecutive_bits_test, "1110001010101001", (0, 1)),
        (longest_sequence_test, "1110001010101001", (0, 1)),
    ],
)
def test_p_value_range(test_function, sequence, expected_range):
    """
    Проверяет, что P-значение, возвращаемое тестируемой функцией,
    находится в ожидаемом диапазоне (0 <= P <= 1).

    Параметры:
        test_function: функция, которая будет протестирована.
        sequence: строка последовательности битов для анализа.
        expected_range: ожидаемый диапазон P-значения.
    """
    p_value = test_function(sequence)
    assert expected_range[0] <= p_value <= expected_range[1], f"{
        test_function.__name__} returned invalid P-value"


def test_frequency_test_empty_sequence():
    """
    Проверяет, что функция frequency_bitwise_test выбрасывает ValueError,
    если ей передать пустую строку.
    """
    with pytest.raises(ValueError):
        frequency_bitwise_test("")


def test_invalid_characters_in_sequence():
    """
    Проверяет, что функция consecutive_bits_test выбрасывает ValueError,
    если последовательность содержит недопустимые символы.
    """
    with pytest.raises(ValueError):
        consecutive_bits_test("11002")


def test_write_text_mocked(monkeypatch):
    """
    Проверяет, что функция write_text вызывает open с правильными аргументами
    и записывает содержимое в файл, используя мокацию функции open.

    monkeypatch используется для подмены стандартной функции open на кастомный объект.
    """
    def mock_open(path, mode, encoding=None):
        assert path == "mocked_file.txt"
        assert mode == "a"
        assert encoding == "utf-8"

        class MockFile:
            def write(self, content):
                assert content == "test content"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockFile()

    monkeypatch.setattr("builtins.open", mock_open)
    assert write_text("mocked_file.txt", "test content") is True


def test_read_json_invalid_file():
    """
    Проверяет, что функция read_json выбрасывает FileNotFoundError,
    если переданный путь указывает на несуществующий файл.
    """
    with pytest.raises(FileNotFoundError):
        read_json("tipitipitipi.json")


def test_read_json_valid():
    """
    Проверяет, что функция read_json корректно читает и парсит JSON-файл,
    создаёт временный файл test.json, записывает в него
    тестовые данные и проверяет корректность их считывания.
    """
    file_path = os.path.join("lab_5", "test.json")
    os.makedirs("lab_5", exist_ok=True)

    valid_json = '{"key": "value"}'
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(valid_json)

    result = read_json(file_path)
    assert result == {"key": "value"}

    os.remove(file_path)


def test_write_and_read_text():
    """
    Проверяет, что функция write_text корректно записывает данные в файл,
    а затем эти данные можно прочитать стандартными средствами.
    """
    file_path = os.path.join("lab_5", "test_write_and_read.txt")
    os.makedirs("lab_5", exist_ok=True)

    content = "This is a test content."

    assert write_text(file_path, content) is True

    with open(file_path, "r", encoding="utf-8") as f:
        read_content = f.read()
    assert read_content == content

    os.remove(file_path)
