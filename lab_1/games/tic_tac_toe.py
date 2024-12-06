class TicTacToe:
    def __init__(self):
        self.board = [" " for _ in range(9)]  # Инициализация пустого поля
        self.current_player = "X"  # Начальный игрок

    def print_board(self):
        """Отображение текущего состояния игрового поля."""
        print(f"{self.board[0]} | {self.board[1]} | {self.board[2]}")
        print("--+---+--")
        print(f"{self.board[3]} | {self.board[4]} | {self.board[5]}")
        print("--+---+--")
        print(f"{self.board[6]} | {self.board[7]} | {self.board[8]}")

    def check_winner(self):
        """Проверка на наличие победителя."""
        winning_combinations = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # Горизонтали
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # Вертикали
            (0, 4, 8),
            (2, 4, 6),  # Диагонали
        ]
        for combo in winning_combinations:
            if (
                self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != " "
            ):
                return self.board[combo[0]]
        return None

    def play(self):
        """Основной игровой процесс."""
        for turn in range(9):
            self.print_board()
            move = int(input(f"Игрок {self.current_player}, выберите позицию (1-9): ")) - 1

            if move < 0 or move > 8 or self.board[move] != " ":
                print("Некорректный ввод или позиция уже занята. Попробуйте снова.")
                continue

            self.board[move] = self.current_player
            winner = self.check_winner()
            if winner:
                self.print_board()
                print(f"Поздравляем! Игрок {winner} выиграл!")
                return

            self.current_player = "O" if self.current_player == "X" else "X"

        self.print_board()
        print("Игра закончилась вничью!")


def tic_tac_toe():
    game = TicTacToe()
    game.play()
