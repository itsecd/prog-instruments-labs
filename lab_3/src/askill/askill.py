from .models import Construction, Action, Canvas
from .parser import Parser


class Askill:
    """Main class for Askill ASCII mini-language"""
    def __init__(self, constructions: list[Construction]):
        if not isinstance(constructions[0], Canvas):
            raise RuntimeError("Canvas should always come first")

        self.canvas = constructions[0]
        self.constructions: list[Action] = constructions[1:]

    @classmethod
    def parse(cls, text: str):
        """Parse input text

        Args:
            text (str): input text

        Returns:
            Askill: new instance of Askill class
        """
        constructions = Parser.parse(text)
        return cls(constructions)

    def draw(self):
        """Draw constructions (fill the matrix)"""
        for construction in self.constructions:
            construction.construct(self.canvas)

    def render(self, spaces: int = 1, enters: int = 1) -> str:
        """Returns the final drawing (сonverts matrix to text)

        Args:
            spaces (int, optional): Number of spaces between lines. Defaults to 1.
            enters (int, optional): Number of enters between lines. Defaults to 1.

        Raises:
            ValueError: spaces < 0
            ValueError: enters < 1

        Returns:
            str: final drawing
        """
        if spaces < 0:
            raise ValueError("Spaces count cannot be < 0")

        if enters < 1:
            raise ValueError("Enters count cannot be < 1")

        return ("\n" * enters).join(
            (" " * spaces).join(row) for row in self.canvas.matrix
        )
