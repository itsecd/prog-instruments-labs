class MazeGame:
    def __init__(self):
        self.mazeLayout = [
            "#########",
            "#       #",
            "# ##### #",
            "# #   # #",
            "# # # # #",
            "#   #   #",
            "#########"
        ]
        self.playerPos = [1, 1]  # Начальная позиция игрока
        self.exitPos = [5, 7]    # Позиция выхода

    def printMaze(self):
        """Отображение лабиринта с игроком."""
        for i in range(len(self.mazeLayout)):
            row = list(self.mazeLayout[i])
            if self.playerPos[0] == i:
                row[self.playerPos[1]] = 'P'  # Отображаем игрока
            print("".join(row))

    def isMoveValid(self, newPos):
        """Проверка, можно ли переместиться на новую позицию."""
        return self.mazeLayout[newPos[0]][newPos[1]] == ' '

    def play(self):
        """Основной игровой процесс."""
        while True:
            self.printMaze()

            if self.playerPos == self.exitPos:
                print("Поздравляю! Вы нашли выход!")
                break

            move = input("Введите ваше движение (W - вверх, S - вниз, A - влево, D - вправо): ").lower()

            if move == "w":  # Вверх
                newPos = [self.playerPos[0] - 1, self.playerPos[1]]
            elif move == "s":  # Вниз
                newPos = [self.playerPos[0] + 1, self.playerPos[1]]
            elif move == "a":  # Влево
                newPos = [self.playerPos[0], self.playerPos[1] - 1]
            elif move == "d":  # Вправо
                newPos = [self.playerPos[0], self.playerPos[1] + 1]
            else:
                print("Неверное движение! Попробуйте снова.")
                continue

            # Проверяем, можно ли переместиться на новую позицию
            if self.isMoveValid(newPos):
                self.playerPos = newPos
            else:
                print("Вы не можете пройти через стену!")

def maze():
    game = MazeGame()
    game.play()