import random
import os

class Game2048:
    def __init__(self):
        self.board = self.initGame()

    def initGame(self):
        """Инициализация игровой доски."""
        board = [[0] * 4 for _ in range(4)]
        self.addNewTile(board)
        self.addNewTile(board)
        return board

    def addNewTile(self, board):
        """Добавление новой плитки (2 или 4) на доску."""
        x, y = random.randint(0, 3), random.randint(0, 3)
        while board[x][y] != 0:
            x, y = random.randint(0, 3), random.randint(0, 3)
        board[x][y] = random.choice([2, 4])

    def printBoard(self):
        """Вывод игровой доски на экран."""
        os.system('cls' if os.name == 'nt' else 'clear')
        for row in self.board:
            print("\t".join(str(num) if num != 0 else "." for num in row))
            print()

    def slideAndMerge(self, row):
        """Сдвиг и объединение плиток в строке."""
        newRow = [num for num in row if num != 0]
        mergedRow = []
        skip = False

        for i in range(len(newRow)):
            if skip:
                skip = False
                continue
            if i < len(newRow) - 1 and newRow[i] == newRow[i + 1]:
                mergedRow.append(newRow[i] * 2)
                skip = True
            else:
                mergedRow.append(newRow[i])

        mergedRow += [0] * (len(row) - len(mergedRow))
        return mergedRow

    def move(self, direction):
        """Перемещение плиток в заданном направлении."""
        board = self.board
        if direction in ('w', 's'):
            board = [list(row) for row in zip(*board)]  # Транспонируем для вертикального движения

        if direction in ('s', 'd'):
            board = [row[::-1] for row in board]  # Реверсируем строки для движения вниз и вправо

        newBoard = []
        for row in board:
            newRow = self.slideAndMerge(row)
            newBoard.append(newRow)

        if direction in ('s', 'd'):
            newBoard = [row[::-1] for row in newBoard]  # Реверсируем обратно

        if direction in ('w', 's'):
            newBoard = [list(row) for row in zip(*newBoard)]  # Транспонируем обратно

        return newBoard

    def isGameOver(self):
        """Проверка, закончилась ли игра."""
        for row in self.board:
            if 2048 in row:
                print("Поздравляем! Вы выиграли!")
                return True
            if 0 in row:
                return False
        for i in range(4):
            for j in range(3):
                if self.board[i][j] == self.board[i][j + 1] or self.board[j][i] == self.board[j + 1][i]:
                    return False
        print("Игра окончена! Попробуйте еще раз.")
        return True

    def play(self):
        """Основная функция игры 2048."""
        while True:
            self.printBoard()
            moveInput = input("Введите направление (w/a/s/d для вверх/влево/вниз/вправо, q для выхода): ").lower()

            if moveInput in ('w', 'a', 's', 'd'):
                newBoard = self.move(moveInput)
                if newBoard != self.board:
                    self.board = newBoard
                    self.addNewTile(self.board)
                if self.isGameOver():
                    break
            elif moveInput == 'q':
                print("Вы вышли из игры.")
                break
            else:
                print("Неверный ввод! Пожалуйста, используйте w/a/s/d или q.")

def game_2048():
    game = Game2048()
    game.play()