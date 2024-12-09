import os
import random
import logging
from typing import List
from .controls import Controls


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Game2048:
    def __init__(self) -> None:
        """Инициализация игры 2048 и создание начальной игровой доски."""
        self.board: List[List[int]] = self.init_game()
        self.controls = Controls()  # Инициализация управления
        logging.info("Игра 2048 начата. Начальная доска:\n%s", self.board)

    def init_game(self) -> List[List[int]]:
        """
        Инициализация игровой доски.

        Returns:
            List[List[int]]: Начальная игровая доска 4x4.
        """
        board = [[0] * 4 for _ in range(4)]
        self.add_new_tile(board)
        self.add_new_tile(board)
        return board

    def add_new_tile(self, board: List[List[int]]) -> None:
        """
        Добавление новой плитки (2 или 4) на доску.

        Args:
            board (List[List[int]]): Игровая доска.
        """
        x, y = random.randint(0, 3), random.randint(0, 3)
        while board[x][y] != 0:
            x, y = random.randint(0, 3), random.randint(0, 3)
        board[x][y] = random.choice([2, 4])
        logging.info("Добавлена новая плитка %d на позицию (%d, %d)", board[x][y], x, y)

    def print_board(self) -> None:
        """Вывод игровой доски на экран."""
        os.system("cls" if os.name == "nt" else "clear")
        for row in self.board:
            print("\t".join(str(num) if num != 0 else "." for num in row))
            print()

    def slide_and_merge(self, row: List[int]) -> List[int]:
        """
        Сдвиг и объединение плиток в строке.

        Args:
            row (List[int]): Строка плиток.

        Returns:
            List[int]: Обновленная строка плиток после сдвига и объединения.
        """
        new_row = [num for num in row if num != 0]
        merged_row = []
        skip = False

        for i in range(len(new_row)):
            if skip:
                skip = False
                continue
            if i < len(new_row) - 1 and new_row[i] == new_row[i + 1]:
                merged_row.append(new_row[i] * 2)
                skip = True
                logging.info("Объединены плитки %d и %d в %d", new_row[i], new_row[i], new_row[i] * 2)
            else:
                merged_row.append(new_row[i])

        merged_row += [0] * (len(row) - len(merged_row))
        return merged_row

    def move(self, direction: str) -> List[List[int]]:
        """
        Перемещение плиток в заданном направлении.

        Args:
            direction (str): Направление движения ('w', 'a', 's', 'd').

        Returns:
            List[List[int]]: Обновленная игровая доска после перемещения.
        """
        board = self.board
        if direction in ("w", "s"):
            board = [list(row) for row in zip(*board)]  # Транспонируем для вертикального движения

        if direction in ("s", "d"):
            board = [row[::-1] for row in board]  # Реверсируем строки для движения вниз и вправо

        new_board = []
        for row in board:
            new_row = self.slide_and_merge(row)
            new_board.append(new_row)

        if direction in ("s", "d"):
            new_board = [row[::-1] for row in new_board]  # Реверсируем обратно

        if direction in ("w", "s"):
            new_board = [list(row) for row in zip(*new_board)]  # Транспонируем обратно

        logging.info("Доска после движения %s:\n%s", direction, new_board)
        return new_board

    def is_game_over(self) -> bool:
        """
        Проверка, закончилась ли игра.

        Returns:
            bool: True, если игра окончена, иначе False.
        """
        for row in self.board:
            if 2048 in row:
                print("Поздравляем! Вы выиграли!")
                logging.info("Игрок выиграл, достигнув 2048.")
                return True
            if 0 in row:
                return False
        for i in range(4):
            for j in range(3):
                if (self.board[i][j] == self.board[i][j + 1] or
                        self.board[j][i] == self.board[j + 1][i]):
                    return False
        print("Игра окончена! Попробуйте еще раз.")
        logging.info("Игра окончена, игрок проиграл.")
        return True

    def play(self) -> None:
        """Основная функция игры 2048."""
        while True:
            self.print_board()
            self.controls.display_controls()
            move_input = input("Введите направление (w/a/s/d для вверх/влево/вниз/вправо, q для выхода): ").lower()

            if move_input in self.controls.key_map:
                new_board = self.move(move_input)
                if new_board != self.board:
                    self.board = new_board
                    self.add_new_tile(self.board)
                if self.is_game_over():
                    break
            elif move_input == "q":
                print("Вы вышли из игры.")
                logging.info("Игрок вышел из игры.")
                break
            else:
                print("Неверный ввод! Пожалуйста, используйте w/a/s/d или q.")
                logging.warning("Пользователь ввел неверное направление: %s", move_input)


def game_2048() -> None:
    """Запуск игры 2048."""
    game = Game2048()
    game.play()
