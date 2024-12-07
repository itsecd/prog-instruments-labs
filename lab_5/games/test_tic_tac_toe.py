import unittest
from unittest.mock import patch
from io import StringIO
from tic_tac_toe import TicTacToe


class TestTicTacToe(unittest.TestCase):

    def setUp(self):
        """Создаем экземпляр игры перед каждым тестом."""
        self.game = TicTacToe()

    def test_initial_board(self):
        """Проверка начального состояния игрового поля."""
        self.assertEqual(self.game.board, [" "] * 9)

    def test_check_winner_x(self):
        """Проверка победителя X."""
        self.game.board = ["X", "X", "X", " ", "O", "O", " ", " ", " "]
        self.assertEqual(self.game.check_winner(), "X")

    def test_check_winner_o(self):
        """Проверка победителя O."""
        self.game.board = ["O", "O", "O", "X", "X", " ", " ", " ", " "]
        self.assertEqual(self.game.check_winner(), "O")

    def test_no_winner(self):
        """Проверка, что нет победителя."""
        self.game.board = ["X", "O", "X", "O", "X", "O", "O", "X", "O"]
        self.assertIsNone(self.game.check_winner())

    @patch('builtins.input', side_effect=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    def test_play_x_wins(self, mock_input):
        """Тестирование игры, где X выигрывает."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.game.play()
            output = fake_out.getvalue()
            self.assertIn("Поздравляем! Игрок X выиграл!", output)

    @patch('builtins.input', side_effect=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    def test_play_draw(self, mock_input):
        """Тестирование игры, которая заканчивается вничью."""
        with patch('builtins.input', side_effect=['1', '2', '3', '4', '5', '6', '7', '8', '9']):
            self.game.board = ["X", "O", "X", "O", "X", "O", "O", "X", "O"]
            with patch('sys.stdout', new=StringIO()) as fake_out:
                self.game.play()
                output = fake_out.getvalue()
                self.assertIn("Игра закончилась вничью!", output)


if __name__ == "__main__":
    unittest.main()
