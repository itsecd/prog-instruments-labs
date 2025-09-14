import re
from ..models import (
    Construction, Canvas,
    Rect, Circle, Line
)


class Parser:
    _PATTERNS = {
        "canvas": r"^\# CANVAS +\"(?P<fill>.)\" +\((?P<width>\d+),(?P<height>\d+)\)\s*\#*$",
        "rect": r"^RECT +\"(?P<border>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)( FILL \"(?P<fill>.)\")?\s*\#*$",
        "circle": r"^CIRCLE +\"(?P<border>.)\" +\((?P<x>\d+),(?P<y>\d+),(?P<r>\d+)\)( FILL \"(?P<fill>.)\")?\s*$",
        "line": r"^LINE +\"(?P<fill>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)\s*$",
    }

    @classmethod
    def _parse_string(cls, string: str) -> Construction:
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
        strings = text.split("\n")

        constructions = []

        canvas = cls._parse_string(strings[0])
        if not isinstance(canvas, Canvas):
            raise RuntimeError("Don't have '# CANVAS' construction in start of file")
        
        constructions.append(canvas)

        for string in strings[1:]:
            if string.isspace() or len(string) == 0:
                continue

            if string.startswith("#"):
                continue
            
            construction = cls._parse_string(string)
            constructions.append(construction)

        return constructions
