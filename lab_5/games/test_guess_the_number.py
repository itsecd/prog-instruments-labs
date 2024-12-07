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

    @patch('builtins.input', side_effect=['abc', 'выход'])
    def test_invalid_input(self, mock_input):
        """Тестирование недопустимого ввода."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.game.play()
            output = fake_out.getvalue().strip()

        self.assertIn("Пожалуйста, введите корректное число.", output)


if __name__ == '__main__':
    unittest.main()
