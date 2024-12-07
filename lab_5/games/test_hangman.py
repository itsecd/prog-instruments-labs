import unittest
from unittest.mock import patch
from hangman import Hangman


class TestHangman(unittest.TestCase):

    def setUp(self):
        """Создание экземпляра игры Hangman перед каждым тестом."""
        self.game = Hangman()

    @patch('builtins.input', side_effect=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    def test_correct_guess(self, mock_input):
        """Тестирование правильного угадывания буквы."""
        self.game.word = "abcde"  # Установим слово для теста
        self.game.play()
        self.assertTrue(self.game.guessed)  # Проверяем, что слово было угадано

    @patch('builtins.input', side_effect=['x', 'y', 'z', 'a', 'b', 'c'])
    def test_incorrect_guess(self, mock_input):
        """Тестирование неправильного угадывания буквы."""
        self.game.word = "abc"  # Установим слово для теста
        self.game.tries = 3  # Установим количество попыток
        self.game.play()
        self.assertFalse(self.game.guessed)  # Проверяем, что слово не было угадано
        self.assertEqual(self.game.tries, 0)  # Проверяем, что попытки закончились

    @patch('builtins.input', side_effect=['a', 'a', 'b', 'c'])
    def test_repeated_guess(self, mock_input):
        """Тестирование повторного угадывания буквы."""
        self.game.word = "abc"  # Установим слово для теста
        self.game.play()
        self.assertEqual(len(self.game.guessed_letters), 3)  # Проверяем, что буква 'a' не была добавлена повторно


if __name__ == '__main__':
    unittest.main()
