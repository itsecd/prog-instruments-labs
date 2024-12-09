import pytest
import unittest
from unittest.mock import patch
from io import StringIO
import random
from guess_the_number import GuessTheNumber


class TestGuessTheNumber(unittest.TestCase):

    def setUp(self):
        """Создание экземпляра игры GuessTheNumber перед каждым тестом."""
        random.seed(0)  # Устанавливаем фиксированное начальное значение для генератора случайных чисел
        self.game = GuessTheNumber()

    @patch('builtins.input', side_effect=['50', 'выход'])
    def test_guessing_game(self, mock_input):
        """Тестирование игрового процесса."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.game.play()
            output = fake_out.getvalue().strip()

        # Проверяем, что игра завершилась
        self.assertIn("Вы вышли из игры.", output)

    @patch('builtins.input', side_effect=['50', 'выход'])
    def test_correct_guess(self, mock_input):
        """Тестирование правильного угадывания числа."""
        self.game.number_to_guess = 50  # Устанавливаем фиксированное число для теста
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.game.play()
            output = fake_out.getvalue().strip()

        self.assertIn("Поздравляем! Вы угадали число 50", output)

    @patch('builtins.input', side_effect=['45', 'выход'])
    def test_hot_guess(self, mock_input):
        """Тестирование "горячего" угадывания."""
        self.game.number_to_guess = 50  # Устанавливаем фиксированное число для теста
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.game.play()
            output = fake_out.getvalue().strip()

        self.assertIn("Очень горячо! Вы очень близки к правильному числу.", output)


@pytest.fixture
def game():
    """Создание экземпляра игры для тестирования."""
    return GuessTheNumber()


def test_initial_number_range(game):
    """Тестирование начального диапазона загаданного числа."""
    assert -100 <= game.number_to_guess <= 100


@pytest.mark.parametrize("guess, expected_output", [
    (45, "Очень горячо! Вы очень близки к правильному числу."),
    (55, "Очень горячо! Вы очень близки к правильному числу."),
    (40, "Горячо! Вы близки к правильному числу."),
    (60, "Горячо! Вы близки к правильному числу."),
    (30, "Тепло. Вы находитесь в пределах 20."),
    (70, "Тепло. Вы находитесь в пределах 20."),
    (20, "Слишком маленькое число. Попробуйте снова."),
])
def test_check_guess_various(game, guess, expected_output):
    """Параметризованный тест для проверки различных угаданных чисел."""
    game.number_to_guess = 50  # Установим число для теста
    game.attempts = 0
    with patch('sys.stdout', new=StringIO()) as fake_out:
        game.check_guess(guess)
        output = fake_out.getvalue().strip()
    assert output == expected_output


def test_invalid_input(game):
    """Тестирование обработки некорректного ввода."""
    with patch('builtins.input', side_effect=["abc", "выход"]), \
         patch('sys.stdout', new=StringIO()) as fake_out:
        game.play()
        output = fake_out.getvalue().strip()
    assert "Пожалуйста, введите корректное число." in output
    assert "Вы вышли из игры." in output


if __name__ == '__main__':
    unittest.main()
