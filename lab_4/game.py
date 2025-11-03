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
        
    def get_next_head(self) -> tuple[int, int]:
        r, c = self.snake[0]
        match self.direction:
            case Direction.UP: r -= 1
            case Direction.DOWN: r += 1
            case Direction.LEFT: c -= 1
            case Direction.RIGHT: c += 1
        return (r, c)

    def collides(self, head: tuple[int, int]) -> bool:
        return (
            head[0] < 0 or head[1] < 0 or
            head[0] >= self.rows or head[1] >= self.cols or
            head in self.snake
        )

    def move_snake(self) -> bool:
        new_head = self.get_next_head()
        if self.collides(new_head):
            self.finished = True
            return False
        self.snake.insert(0, new_head)
        if new_head in self.food:
            self.food.remove(new_head)
        else:
            self.snake.pop()
        return True