from .construction import Context


class Canvas(Context):
    def __init__(self, width: int, height: int, fill: str):
        if width <= 0:
            raise ValueError("width cannot be <= 0")

        if height <= 0:
            raise ValueError("height cannot be <= 0")

        self.fill = fill
        self.width = width
        self.height = height

        self.matrix: list[list[str]] = [[fill] * width for _ in range(height)]

    def __repr__(self):
        return f"Canvas({self.width}, {self.height}, {repr(self.fill)})"

    def construct(self, *args, **kwargs):
        pass        
