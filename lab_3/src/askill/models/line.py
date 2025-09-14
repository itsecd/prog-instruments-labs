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
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2

        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy

        while True:
            if 0 <= y1 < canvas.height and 0 <= x1 < canvas.width:
                canvas.matrix[y1][x1] = self.fill

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy
