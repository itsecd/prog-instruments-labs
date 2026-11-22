import pytest
from unittest.mock import patch, mock_open
from generators.python.generator import generateDRV, generateDRVec, writeDRVec, argument_parsing

# --- ПРИМИТИВНЫЕ ТЕСТЫ ---


def test_generateDRV():
    # Проверяем, что генератор выдает только 0 или 1
    for _ in range(100):  # Запускаем 100 раз для надежности
        val = generateDRV()
        assert val in (0, 1)


def test_generateDRVec_length():
    # Проверяем, что вектор имеет правильную длину (128)
    vec = generateDRVec()
    assert len(vec) == 128
    assert all(item in (0, 1) for item in vec)

# --- ПРОДВИНУТЫЙ ТЕСТ №4: ТЕСТИРОВАНИЕ CLI АРГУМЕНТОВ ---


@patch("sys.argv", ["generator.py", "test_output.txt"])
def test_argument_parsing():
    # Подменяем аргументы командной строки
    result = argument_parsing()
    assert result == ["test_output.txt"]

# --- ПРОДВИНУТЫЙ ТЕСТ №5: МОК ДЛЯ ЗАПИСИ В ФАЙЛ ---


@patch("builtins.open", new_callable=mock_open)
def test_writeDRVec(mock_file):
    # Подменяем open, чтобы не создавать реальный файл
    output_path = "fake_path.txt"
    writeDRVec(output_path)

    # Проверяем, что файл был открыт в режиме записи
    mock_file.assert_called_once_with(output_path, "+w")

    # Проверяем, что метод write был вызван (строка была записана)
    handle = mock_file()
    handle.write.assert_called_once()
    # Можно даже проверить, что записана строка из 128 символов
    written_string = handle.write.call_args[0][0]
    assert len(written_string) == 128
