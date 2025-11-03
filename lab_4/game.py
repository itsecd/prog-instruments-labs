from enum import Enum

class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class GameBoard:
    """Game state and logic for Snake."""
    def __init__(self, rows: int, cols: int, size: int = 2):
        self.rows = rows
        self.cols = cols
        self.size = size
        self.finished = False
        self.board = [[" " for _ in range(cols)] for _ in range(rows)]
        self.snake = [((rows // 2) + i, cols // 2) for i in range(size)]
        self.food: list[tuple[int, int]] = []
        self.direction = Direction.LEFT