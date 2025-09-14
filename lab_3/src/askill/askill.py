from .models import Construction, Action, Canvas
from .parser import Parser


class Askill:
    def __init__(self, constructions: list[Construction]):
        if not isinstance(constructions[0], Canvas):
            raise RuntimeError("Canvas should always come first")

        self.canvas = constructions[0]
        self.constructions: list[Action] = constructions[1:]

    @classmethod
    def parse(cls, text: str):
        constructions = Parser.parse(text)
        return cls(constructions)

    def draw(self):
        for construction in self.constructions:
            construction.construct(self.canvas)

    def render(self, spaces: int = 1, enters: int = 1) -> str:
        if spaces < 0:
            raise ValueError("Spaces count cannot be < 0")

        if enters < 1:
            raise ValueError("Enters count cannot be < 0")

        return ("\n" * enters).join(
            (" " * spaces).join(row) for row in self.canvas.matrix
        )
