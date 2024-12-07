import logging
from typing import Optional, List


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TicTacToe:
    """Класс для игры в Крестики-Нолики."""

    def __init__(self) -> None:
        """Инициализация пустого поля и текущего игрока."""
        self.board: List[str] = [" " for _ in range(9)]  # Инициализация пустого поля
        self.current_player: str = "X"  # Начальный игрок
        logging.info("Игра в Крестики-Нолики начата. Начальный игрок: %s", self.current_player)

    def print_board(self) -> None:
        """Отображение текущего состояния игрового поля."""
        print(f"{self.board[0]} | {self.board[1]} | {self.board[2]}")
        print("--+---+--")
        print(f"{self.board[3]} | {self.board[4]} | {self.board[5]}")
        print("--+---+--")
        print(f"{self.board[6]} | {self.board[7]} | {self.board[8]}")

    def check_winner(self) -> Optional[str]:
        """
        Проверка на наличие победителя.

        Returns:
            str | None: Возвращает 'X' или 'O' в случае победы, иначе None.
        """
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Горизонтали
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Вертикали
            (0, 4, 8), (2, 4, 6),              # Диагонали
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != " ":
                logging.info("Игрок %s выиграл!", self.board[combo[0]])
                return self.board[combo[0]]
        return None

    def play(self) -> None:
        """Основной игровой процесс."""
        for turn in range(9):
            self.print_board()
            move = input(f"Игрок {self.current_player}, выберите позицию (1-9): ")

            # Проверка на корректность ввода
            if not move.isdigit() or not (1 <= int(move) <= 9):
                print("Некорректный ввод. Пожалуйста, введите число от 1 до 9.")
                logging.warning("Игрок %s ввел некорректное значение: %s", self.current_player, move)
                continue

            move = int(move) - 1

            if self.board[move] != " ":
                print("Позиция уже занята. Попробуйте снова.")
                logging.warning("Игрок %s пытается занять уже занятую позицию: %d", self.current_player, move + 1)
                continue

            self.board[move] = self.current_player
            logging.info("Игрок %s поставил знак на позицию %d", self.current_player, move + 1)

            winner = self.check_winner()
            if winner:
                self.print_board()
                print(f"Поздравляем! Игрок {winner} выиграл!")
                return

            self.current_player = "O" if self.current_player == "X" else "X"

        self.print_board()
        print("Игра закончилась вничью!")
        logging.info("Игра закончилась вничью.")


def tic_tac_toe() -> None:
    """Запуск игры в Крестики-Нолики."""
    game = TicTacToe()
    game.play()
