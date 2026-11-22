import pytest
from unittest.mock import mock_open, patch
from filereading import read_file, read_vector, read_sequence

# --- ПРИМИТИВНЫЕ ТЕСТЫ ---


def test_read_vector_valid():
    # Тест нормального поведения парсера
    assert read_vector("101") == [True, False, True]
    assert read_vector("") == []

# --- ПРОДВИНУТЫЙ ТЕСТ №2: МОКИ И ИСКЛЮЧЕНИЯ ---


def test_read_vector_invalid():
    # Согласно лекции: "Возможно, выброс исключений. Недопустимые данные"
    with pytest.raises(ValueError, match="Uncorrect file format!"):
        read_vector("102")  # Символ '2' недопустим


@patch("builtins.open", new_callable=mock_open, read_data="1010\n")
def test_read_file_mock(mock_file):
    # Использование мока для функции open.
    # Мы не создаем реальный файл, а "подменяем" его.
    result = read_file("dummy_path.txt")

    # Проверяем, что open был вызван с правильными аргументами
    mock_file.assert_called_once_with("dummy_path.txt", "r+")
    # Проверяем, что вернулось правильное значение (без \n)
    assert result == "1010"

# --- ПРОДВИНУТЫЙ ТЕСТ №3: КОМБИНАЦИЯ МОКА И ПАРАМЕТРИЗАЦИИ ---


@pytest.mark.parametrize("file_content, expected_result", [
    ("1\n", [True]),
    ("010\n", [False, True, False]),
])
@patch("filereading.read_file")
def test_read_sequence_mock(mock_read_file, file_content, expected_result):
    # Подменяем read_file, чтобы не зависеть от файловой системы
    mock_read_file.return_value = file_content.strip()

    assert read_sequence("any_path.txt") == expected_result
