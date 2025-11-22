import sys


class TicTacToe:
    def __init__(self):
        self.board = [["", "", ""] for _ in range(3)]
        self.used_positions = []
        self.player_symbol = ""
        self.computer_symbol = ""
        self.current_player = ""

    def print_board(self):
        """Print the current game board"""
        for i, row in enumerate(self.board):
            print(" | ".join(cell if cell else " " for cell in row))
            if i < 2:
                print("---------")

    def setup_game(self):
        """Initialize game settings"""
        print("Welcome to the two player mode of Tic-Tac-Toe")
        print("Press 1 to be O and 2 to be X")

        choice = int(input("Enter here: "))
        if choice == 1:
            self.player_symbol = "O"
            self.computer_symbol = "X"
        else:
            self.player_symbol = "X"
            self.computer_symbol = "O"

        print("Press 1 to play first, press 2 to let computer play first")
        first_player = int(input("Enter here: "))
        self.current_player = "player" if first_player == 1 else "computer"

    def get_position(self, row, col):
        """Calculate position number from row and column"""
        return row * 3 + col + 1

    def make_move(self, position, symbol):
        """Place symbol on the board at given position"""
        if position < 1 or position > 9:
            return False

        if position in self.used_positions:
            return False

        row = (position - 1) // 3
        col = (position - 1) % 3

        self.board[row][col] = symbol
        self.used_positions.append(position)
        return True

    def check_winner(self, symbol):
        """Check if the given symbol has won"""
        board = self.board

        # Check rows
        for row in board:
            if all(cell == symbol for cell in row):
                return True

        # Check columns
        for col in range(3):
            if all(board[row][col] == symbol for row in range(3)):
                return True

        # Check diagonals
        if all(board[i][i] == symbol for i in range(3)):
            return True
        if all(board[i][2-i] == symbol for i in range(3)):
            return True

        return False

    def is_board_full(self):
        """Check if the board is completely filled"""
        return all(all(cell != "" for cell in row) for row in self.board)

    def game_over(self):
        """Handle game conclusion"""
        choice = int(input("Enter 1 to terminate the game: "))
        if choice == 1:
            sys.exit()

    def player_turn(self):
        """Handle player's turn"""
        print("\nPLAYER TURN")
        while True:
            try:
                position = int(input("Enter position (1-9): "))
                if self.make_move(position, self.player_symbol):
                    break
                else:
                    print("Invalid position or position already taken. Try again.")
            except ValueError:
                print("Please enter a valid number.")

        self.print_board()

        if self.check_winner(self.player_symbol):
            print("PLAYER HAS WON!")
            self.game_over()
            return True
        elif self.is_board_full():
            print("The game has been a tie!")
            self.game_over()
            return True

        return False

    def computer_turn(self):
        """Handle computer's turn with strategic moves"""
        print("\nCOMPUTER TURN")

        # Try to find winning move
        move = self.find_winning_move(self.computer_symbol)
        if not move:
            # Block player's winning move
            move = self.find_winning_move(self.player_symbol)

        # Strategic moves
        if not move:
            move = self.make_strategic_move()

        self.make_move(move, self.computer_symbol)
        print(f"Computer's move is {move}")
        self.print_board()

        if self.check_winner(self.computer_symbol):
            print("COMPUTER HAS WON!")
            self.game_over()
            return True
        elif self.is_board_full():
            print("The game has been a tie!")
            self.game_over()
            return True

        return False

    def find_winning_move(self, symbol):
        """Find a move that would result in a win for the given symbol"""
        for position in range(1, 10):
            if position not in self.used_positions:
                # Test this move
                row = (position - 1) // 3
                col = (position - 1) % 3

                # Save current state
                original_value = self.board[row][col]
                self.board[row][col] = symbol

                # Check if this move wins
                if self.check_winner(symbol):
                    self.board[row][col] = original_value
                    return position

                # Restore original state
                self.board[row][col] = original_value

        return None

    def make_strategic_move(self):
        """Make a strategic move based on game situation"""
        # Center is most valuable
        if 5 not in self.used_positions:
            return 5

        # Corners are next best
        corners = [1, 3, 7, 9]
        available_corners = [pos for pos in corners if pos not in self.used_positions]
        if available_corners:
            return available_corners[0]

        # Finally, edges
        edges = [2, 4, 6, 8]
        available_edges = [pos for pos in edges if pos not in self.used_positions]
        if available_edges:
            return available_edges[0]

        # Fallback - should not happen if board isn't full
        for position in range(1, 10):
            if position not in self.used_positions:
                return position

        return None

    def play_game(self):
        """Main game loop"""
        self.setup_game()
        self.print_board()

        while True:
            if self.current_player == "player":
                if self.player_turn():
                    break
                self.current_player = "computer"
            else:
                if self.computer_turn():
                    break
                self.current_player = "player"


if __name__ == "__main__":
    game = TicTacToe()
    game.play_game()
