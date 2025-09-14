from .construction import Action
from .canvas import Canvas


class Line(Action):
    def __init__(
        self, x1: int, y1: int, x2: int, y2: int,
        fill: str,
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.fill = fill
    
    def __repr__(self):
        return f"Line({self.x1}, {self.y1}, {self.x2}, {self.y2}, {repr(self.fill)})"

    def construct(self, canvas: Canvas):
        canvas.matrix[2][0] = "line"
