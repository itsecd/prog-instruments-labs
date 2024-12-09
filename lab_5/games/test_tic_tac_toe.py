import pytest
import unittest
from unittest.mock import patch
from io import StringIO
from tic_tac_toe import TicTacToe


class TestTicTacToe(unittest.TestCase):

    def setUp(self):
        """Создаем экземпляр игры перед каждым тестом."""
        self.game = TicTacToe()

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


@pytest.fixture
def game():
    """Фикстура для создания экземпляра игры TicTacToe перед каждым тестом."""
    return TicTacToe()


def test_init(game):
    """Проверка начального состояния игрового поля."""
    assert game.board == [" "] * 9
    assert game.current_player == "X"


@pytest.mark.parametrize("move, expected", [
    ("1", 0),
    ("2", 1),
    ("3", 2),
    ("4", 3),
    ("5", 4),
    ("6", 5),
    ("7", 6),
    ("8", 7),
    ("9", 8),
])
def test_move(game, move, expected):
    """Проверка правильности выполнения ходов в игре."""
    with patch("builtins.input", return_value=move):
        game.board = [" "] * 9
        game.current_player = "X"
        game.play()
        assert game.board[expected] == "X"


def test_play(game):
    """Проверка основного игрового процесса."""
    with patch("builtins.input", side_effect=["1", "2", "3", "4", "5", "6", "7", "8", "9"]):
        with patch("builtins.print") as mock_print:
            game.play()
            assert mock_print.call_count > 0


if __name__ == "__main__":
    unittest.main()
