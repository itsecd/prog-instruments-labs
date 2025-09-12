class Surface:
    def __init__(self, width: int, height: int, fill: str = ""):
        self.data: list[list[str]] = [[fill] * width for _ in range(height)]
        self.width = width
        self.height = height
