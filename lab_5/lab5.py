import sys
import logging
from datetime import datetime


class TicTacToe:
    def __init__(self):
        self.board = [["", "", ""] for _ in range(3)]
        self.used_positions = []
        self.player_symbol = ""
        self.computer_symbol = ""
        self.current_player = ""
        self.setup_logging()

    def setup_logging(self):
        """Registering the setting in a file"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'tic_tac_toe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("НАЧАЛО НОВОЙ ИГРЫ В КРЕСТИКИ-НОЛИКИ")

    def print_board(self):
        """Print the current game board"""
        board_str = ""
        for i, row in enumerate(self.board):
            board_str += " | ".join(cell if cell else " " for cell in row) + "\n"
            if i < 2:
                board_str += "---------\n"
        print(board_str)
        self.logger.info(f"Текущее состояние доски:\n{board_str}")

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

        self.logger.info(f"Игрок выбрал символ: {self.player_symbol}, компьютер: {self.computer_symbol}")

        print("Press 1 to play first, press 2 to let computer play first")
        first_player = int(input("Enter here: "))
        self.current_player = "player" if first_player == 1 else "computer"

        self.logger.info(f"Первым ходит: {self.current_player}")

    def get_position(self, row, col):
        """Calculate position number from row and column"""
        return row * 3 + col + 1

    def make_move(self, position, symbol):
        """Place symbol on the board at given position"""
        if position < 1 or position > 9:
            self.logger.warning(f"Попытка неверного хода: позиция {position} вне диапазона 1-9")
            return False

        if position in self.used_positions:
            self.logger.warning(f"Попытка хода в занятую позицию: {position}")
            return False

        row = (position - 1) // 3
        col = (position - 1) % 3

        self.board[row][col] = symbol
        self.used_positions.append(position)

        player_type = "игрок" if symbol == self.player_symbol else "компьютер"
        self.logger.info(f"{player_type} поставил '{symbol}' на позицию {position} (ряд {row + 1}, колонка {col + 1})")
        self.logger.info(f"Занятые позиции: {sorted(self.used_positions)}")

        return True

    def check_winner(self, symbol):
        """Check if the given symbol has won"""
        board = self.board

        # Check rows
        for i, row in enumerate(board):
            if all(cell == symbol for cell in row):
                self.logger.info(f"Победа {symbol}! Выигрышная строка: {i + 1}")
                return True

        # Check columns
        for col in range(3):
            if all(board[row][col] == symbol for row in range(3)):
                self.logger.info(f"Победа {symbol}! Выигрышная колонка: {col + 1}")
                return True

        # Check diagonals
        if all(board[i][i] == symbol for i in range(3)):
            self.logger.info(f"Победа {symbol}! Выигрышная главная диагональ")
            return True
        if all(board[i][2 - i] == symbol for i in range(3)):
            self.logger.info(f"Победа {symbol}! Выигрышная побочная диагональ")
            return True

        return False

    def is_board_full(self):
        """Check if the board is completely filled"""
        is_full = all(all(cell != "" for cell in row) for row in self.board)
        if is_full:
            self.logger.info("Доска полностью заполнена - ничья!")
        return is_full

    def game_over(self):
        """Handle game conclusion"""
        self.logger.info("Игра завершена. Ожидание решения игрока о выходе.")
        choice = int(input("Enter 1 to terminate the game: "))
        if choice == 1:
            self.logger.info("Игрок выбрал завершение программы")
            sys.exit()

    def player_turn(self):
        """Handle player's turn"""
        self.logger.info("--- ХОД ИГРОКА ---")
        print("\nPLAYER TURN")

        while True:
            try:
                position = int(input("Enter position (1-9): "))
                self.logger.info(f"Игрок ввел позицию: {position}")

                if self.make_move(position, self.player_symbol):
                    break
                else:
                    print("Invalid position or position already taken. Try again.")
                    self.logger.warning(f"Игрок ввел недопустимую позицию: {position}")
            except ValueError:
                print("Please enter a valid number.")
                self.logger.error("Игрок ввел нечисловое значение")

        self.print_board()

        if self.check_winner(self.player_symbol):
            self.logger.info("!!! ИГРОК ПОБЕДИЛ !!!")
            print("PLAYER HAS WON!")
            self.game_over()
            return True
        elif self.is_board_full():
            self.logger.info("!!! НИЧЬЯ !!!")
            print("The game has been a tie!")
            self.game_over()
            return True

        return False

    def computer_turn(self):
        """Handle computer's turn with strategic moves"""
        self.logger.info("--- ХОД КОМПЬЮТЕРА ---")
        print("\nCOMPUTER TURN")

        # Try to find winning move
        move = self.find_winning_move(self.computer_symbol)
        move_type = "выигрышный"

        if not move:
            # Block player's winning move
            move = self.find_winning_move(self.player_symbol)
            move_type = "блокирующий"

        # Strategic moves
        if not move:
            move = self.make_strategic_move()
            move_type = "стратегический"

        self.logger.info(f"Компьютер выбрал {move_type} ход на позицию {move}")
        self.make_move(move, self.computer_symbol)
        print(f"Computer's move is {move}")
        self.print_board()

        if self.check_winner(self.computer_symbol):
            self.logger.info("!!! КОМПЬЮТЕР ПОБЕДИЛ !!!")
            print("COMPUTER HAS WON!")
            self.game_over()
            return True
        elif self.is_board_full():
            self.logger.info("!!! НИЧЬЯ !!!")
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
        try:
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
        except Exception as e:
            self.logger.error(f"Произошла ошибка во время игры: {str(e)}", exc_info=True)
            raise
        finally:
            self.logger.info("!!! ИГРА ЗАВЕРШЕНА !!!\n")


if __name__ == "__main__":
    game = TicTacToe()
    game.play_game()