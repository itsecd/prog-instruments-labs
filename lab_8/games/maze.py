from typing import List

from .controls import Controls


class MazeGame:
    def __init__(self) -> None:
        """Инициализация игры в лабиринт."""
        self.maze_layout: List[str] = [
            "#########",
            "#       #",
            "# ##### #",
            "# #   # #",
            "# # # # #",
            "#   #   #",
            "#########",
        ]
        self.player_pos: List[int] = [1, 1]  # Начальная позиция игрока
        self.exit_pos: List[int] = [5, 7]  # Позиция выхода
        self.controls = Controls()  # Инициализация управления

    def print_maze(self) -> None:
        """Отображение лабиринта с игроком."""
        for i in range(len(self.maze_layout)):
            row = list(self.maze_layout[i])
            if self.player_pos[0] == i:
                row[self.player_pos[1]] = "P"  # Отображаем игрока
            print("".join(row))

    def is_move_valid(self, new_pos: List[int]) -> bool:
        """
        Проверка, можно ли переместиться на новую позицию.

        Args:
            new_pos (List[int]): Новая позиция игрока.

        Returns:
            bool: True, если движение допустимо, иначе False.
        """
        return self.maze_layout[new_pos[0]][new_pos[1]] == " "

    def play(self) -> None:
        """Основной игровой процесс."""
        while True:
            self.print_maze()
            self.controls.display_controls()  # Вывод управления

            if self.player_pos == self.exit_pos:
                print("Поздравляю! Вы нашли выход!")
                break

            move = input("Введите ваше движение (W - вверх, S - вниз, A - влево, D - вправо): ").lower()
            direction = self.controls.get_direction(move)  # Получаем направление

            if direction == 'up':
                new_pos = [self.player_pos[0] - 1, self.player_pos[1]]
            elif direction == 'down':
                new_pos = [self.player_pos[0] + 1, self.player_pos[1]]
            elif direction == 'left':
                new_pos = [self.player_pos[0], self.player_pos[1] - 1]
            elif direction == 'right':
                new_pos = [self.player_pos[0], self.player_pos[1] + 1]
            else:
                print("Неверное движение! Попробуйте снова.")
                continue

            # Проверяем, можно ли переместиться на новую позицию
            if self.is_move_valid(new_pos):
                self.player_pos = new_pos
            else:
                print("Вы не можете пройти через стену!")


def maze() -> None:
    """Запуск игры в лабиринт."""
    game = MazeGame()
    game.play()
