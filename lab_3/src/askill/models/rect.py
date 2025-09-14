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
        canvas.matrix[0][0] = "rect"
