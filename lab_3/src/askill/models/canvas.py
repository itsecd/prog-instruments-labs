from .construction import Construction


class Canvas(Construction):
    def __init__(self, width: int, height: int, fill: str):
        if width <= 0:
            raise ValueError("width cannot be <= 0")

        if height <= 0:
            raise ValueError("height cannot be <= 0")

        self.fill = fill
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Canvas({self.width}, {self.height}, {repr(self.fill)})"

    def construct(self, where: list[list[str]]):
        for row in range(self.height):
            for col in range(self.width):
                where[row][col] = self.fill
