from .construction import Action
from .canvas import Canvas


class Rect(Action):
    def __init__(
        self, x1: int, y1: int, x2: int, y2: int,
        border: str, fill: str | None = None,
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.border = border
        self.fill = fill

    def __repr__(self):
        return f"Rect({self.x1}, {self.y1}, {self.x2}, {self.y2}, {repr(self.border)}, {repr(self.fill)})"

    def construct(self, canvas: Canvas):
        for row in range(self.y1, self.y2 + 1):
            for col in range(self.x1, self.x2 + 1):
                if row < 0 or col < 0 or row >= canvas.height or col >= canvas.height:
                    continue

                if row == self.y1 or row == self.y2 or col == self.x1 or col == self.x2:
                    canvas.matrix[row][col] = self.border

                elif self.fill is not None:
                    canvas.matrix[row][col] = self.fill
