from .construction import Action
from .canvas import Canvas


class Circle(Action):
    def __init__(
        self, x: int, y: int, r: int,
        border: str, fill: str | None = None,
    ):
        if r <= 0:
            raise ValueError("Radius r cannot be <= 0")

        self.x = x
        self.y = y
        self.r = r
        self.border = border
        self.fill = fill

    def __repr__(self):
        return f"Circle({self.x}, {self.y}, {self.r}, {repr(self.border)}, {repr(self.fill)})"

    def construct(self, canvas: Canvas):
        canvas.matrix[1][0] = "circle"
