from unittest.mock import patch, mock_open
import generator_analysis


@patch("generator_analysis.analyze_sequence")
@patch("generator_analysis.read_sequence")
@patch("builtins.open", new_callable=mock_open)
def test_main_execution(mock_file, mock_read_seq, mock_analyze):
    # Настраиваем мок для read_sequence
    mock_read_seq.return_value = [True, False, True]

    # ВЫЗЫВАЕМ функцию main() из вашего файла
    generator_analysis.main()

    # Проверяем, что read_sequence вызывался 3 раза (для C++, Java, Python)
    assert mock_read_seq.call_count == 3

    # Проверяем, что analyze_sequence вызывался 3 раза
    assert mock_analyze.call_count == 3

    # Проверяем, что файл был открыт для очистки
    mock_file.assert_any_call("statistics.txt", "w+")
