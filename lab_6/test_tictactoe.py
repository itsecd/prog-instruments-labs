import pytest
from unittest.mock import patch
from io import StringIO

from tictactoe import TicTacToe


class TestTicTacToe:
    def test_initialization(self):
        """Test that game initializes with empty board"""
        game = TicTacToe()
        assert game.board == [["", "", ""], ["", "", ""], ["", "", ""]]
        assert game.used_positions == []
        assert game.player_symbol == ""
        assert game.computer_symbol == ""
        assert game.current_player == ""

    def test_get_position(self):
        """Test position calculation from row and column"""
        game = TicTacToe()
        assert game.get_position(0, 0) == 1
        assert game.get_position(0, 1) == 2
        assert game.get_position(0, 2) == 3
        assert game.get_position(1, 0) == 4
        assert game.get_position(2, 2) == 9

    @pytest.mark.parametrize("position,symbol,expected", [
        (5, "O", True),
        (9, "X", True),
        (10, "X", False),  # Invalid position
        (0, "X", False),   # Invalid position
        (1, "X", False),   # Already used position
    ])
    def test_make_move(self, position, symbol, expected):
        """Test making moves with various positions"""
        game = TicTacToe()
        # First move should work
        result1 = game.make_move(1, "X")
        assert result1 == True
        assert game.board[0][0] == "X"
        assert 1 in game.used_positions

        # Test the parameterized cases
        result = game.make_move(position, symbol)
        assert result == expected

    def test_check_winner_rows(self):
        """Test winner detection in rows"""
        game = TicTacToe()
        # Set up a winning row
        game.board = [
            ["X", "X", "X"],
            ["", "", ""],
            ["", "", ""]
        ]
        assert game.check_winner("X") == True
        assert game.check_winner("O") == False

    def test_check_winner_columns(self):
        """Test winner detection in columns"""
        game = TicTacToe()
        # Set up a winning column
        game.board = [
            ["O", "", ""],
            ["O", "", ""],
            ["O", "", ""]
        ]
        assert game.check_winner("O") == True

    def test_check_winner_diagonals(self):
        """Test winner detection in diagonals"""
        game = TicTacToe()
        # Set up winning main diagonal
        game.board = [
            ["X", "", ""],
            ["", "X", ""],
            ["", "", "X"]
        ]
        assert game.check_winner("X") == True

    def test_is_board_full(self):
        """Test board full detection"""
        game = TicTacToe()
        assert game.is_board_full() == False

        game.board = [
            ["X", "O", "X"],
            ["O", "X", "O"],
            ["O", "X", "O"]
        ]
        assert game.is_board_full() == True

    def test_find_winning_move(self):
        """Test finding winning moves"""
        game = TicTacToe()
        # Set up almost winning board for X
        game.board = [
            ["X", "X", ""],
            ["", "", ""],
            ["", "", ""]
        ]
        game.used_positions = [1, 2]

        winning_move = game.find_winning_move("X")
        assert winning_move == 3

    @patch('builtins.input', side_effect=[1, 1])  # Player chooses O and goes first
    def test_setup_game_player_first(self, mock_input):
        """Test game setup with player going first"""
        game = TicTacToe()
        game.setup_game()

        assert game.player_symbol == "O"
        assert game.computer_symbol == "X"
        assert game.current_player == "player"

    @patch('builtins.input', side_effect=[2, 2])  # Player chooses X and computer goes first
    def test_setup_game_computer_first(self, mock_input):
        """Test game setup with computer going first"""
        game = TicTacToe()
        game.setup_game()

        assert game.player_symbol == "X"
        assert game.computer_symbol == "O"
        assert game.current_player == "computer"

    @patch('builtins.input', side_effect=[4])  # Valid position
    def test_player_turn_valid_move(self, mock_input):
        """Test player turn with valid move"""
        game = TicTacToe()
        game.player_symbol = "X"
        game.computer_symbol = "O"

        with patch('sys.stdout', new_callable=StringIO):
            result = game.player_turn()

        assert game.board[1][0] == "X"  # Position 4 should be row 1, col 0
        assert 4 in game.used_positions
        assert result == False  # Game shouldn't end after one move

    @patch('builtins.input', side_effect=['invalid', '5'])  # Invalid then valid input
    def test_player_turn_invalid_then_valid_input(self, mock_input):
        """Test player turn handling invalid input"""
        game = TicTacToe()
        game.player_symbol = "X"

        with patch('sys.stdout', new_callable=StringIO):
            result = game.player_turn()

        assert game.board[1][1] == "X"  # Position 5 should be set
        assert result == False

    def test_make_strategic_move(self):
        """Test computer's strategic move selection"""
        game = TicTacToe()

        # Empty board - should choose center
        assert game.make_strategic_move() == 5

        # Center taken - should choose corner
        game.make_move(5, "X")
        strategic_move = game.make_strategic_move()
        assert strategic_move in [1, 3, 7, 9]

        # Center and corners taken - should choose edge
        for corner in [1, 3, 7, 9]:
            game.make_move(corner, "X")
        strategic_move = game.make_strategic_move()
        assert strategic_move in [2, 4, 6, 8]

    @patch('builtins.input', side_effect=[1])
    @patch('sys.exit')
    def test_game_over(self, mock_exit, mock_input):
        """Test game over functionality"""
        game = TicTacToe()
        game.game_over()
        mock_exit.assert_called_once()

    def test_computer_turn_winning_move(self):
        """Test computer finds and makes winning move"""
        game = TicTacToe()
        game.computer_symbol = "X"
        # Set up almost winning board for computer
        game.board = [
            ["X", "X", ""],
            ["", "", ""],
            ["", "", ""]
        ]
        game.used_positions = [1, 2]

        with patch('builtins.input', side_effect=[1]), \
                patch('sys.stdout', new_callable=StringIO), \
                patch('sys.exit') as mock_exit:
            result = game.computer_turn()

        assert game.board[0][2] == "X"
        assert result == True
        mock_exit.assert_called_once()

    @pytest.mark.parametrize("board_state,expected_moves", [
        ([  # Block player win
             ["O", "O", ""],
             ["", "X", ""],
             ["", "", "X"]
         ], [3]),
        ([  # Prefer center
             ["", "", ""],
             ["", "", ""],
             ["", "", ""]
         ], [5]),
    ])
    def test_computer_strategic_moves_parametrized(self, board_state, expected_moves):
        """Parameterized test for computer strategic moves"""
        game = TicTacToe()
        game.computer_symbol = "X"
        game.player_symbol = "O"
        game.board = board_state

        # Set used positions based on board state
        game.used_positions = []
        for row in range(3):
            for col in range(3):
                if board_state[row][col]:
                    game.used_positions.append(game.get_position(row, col))

        strategic_move = game.make_strategic_move()
        assert strategic_move in expected_moves
