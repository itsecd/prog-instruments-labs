import re
from ..models import (
    Construction, Canvas,
    Rect, Circle, Line
)
from .patterns import PATTERNS


class Parser:
    """Parser class for Askill"""
    _PATTERNS = PATTERNS

    @classmethod
    def _parse_string(cls, string: str) -> Construction:
        """Method for string parsing

        Args:
            string (str): input string

        Raises:
            RuntimeError: bad parsing

        Returns:
            Construction: result construction
        """
        for command, pattern in cls._PATTERNS.items():
            matches = re.match(pattern, string)
            if matches is None:
                continue

            group_dict = matches.groupdict()

            if command == "rect":
                return Rect(
                    int(group_dict["x1"]), int(group_dict["y1"]),
                    int(group_dict["x2"]), int(group_dict["y2"]),
                    group_dict["border"], group_dict["fill"],
                )
            elif command == "circle":
                return Circle(
                    int(group_dict["x"]), int(group_dict["y"]),
                    int(group_dict["r"]),
                    group_dict["border"], group_dict["fill"],
                )
            elif command == "line":
                return Line(
                    int(group_dict["x1"]), int(group_dict["y1"]),
                    int(group_dict["x2"]), int(group_dict["y2"]),
                    group_dict["fill"],
                )
            elif command == "canvas":
                return Canvas(
                    int(group_dict["width"]), int(group_dict["height"]),
                    group_dict["fill"],
                )

        raise RuntimeError(f"Bad parsing: {string}")

    @classmethod
    def parse(cls, text: str) -> list[Construction]:
        """Method for text parsing

        Args:
            text (str): input text

        Raises:
            RuntimeError: there is no "# Canvas" construct at the beginning of the file

        Returns:
            list[Construction]: list of result constructions
        """
        strings = text.split("\n")

        constructions = []

        canvas = cls._parse_string(strings[0])
        if not isinstance(canvas, Canvas):
            raise RuntimeError("There is no \"# Canvas\" \
                               construct at the beginning of the file")

        constructions.append(canvas)

        for string in strings[1:]:
            if string.isspace() or len(string) == 0:
                continue

            if string.startswith("#"):
                continue

            construction = cls._parse_string(string)
            constructions.append(construction)

        return constructions
