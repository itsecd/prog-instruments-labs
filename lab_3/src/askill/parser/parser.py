import re
from ..models import Construction, Rect, Canvas


class Parser:
    _PATTERNS = {
        "canvas": r"^\# (?P<command>CANVAS) +\"(?P<fill>.)\" +\((?P<width>\d+),(?P<height>\d+)\)\s*\#*$",
        "rect": r"^(?P<command>RECT) +\"(?P<border>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)( FILL \"(?P<fill>.)\")?\s*\#*$",
        "circle": r"^(?P<cmd>CIRCLE) +\"(?P<border>.)\" +\((?P<x>\d+),(?P<y>\d+),(?P<r>\d+)\)( FILL \"(?P<fill>.)\")?\s*$",
        "line": r"^(?P<cmd>LINE) +\"(?P<fill>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)\s*$",
    }

    @classmethod
    def _parse_string(cls, string: str) -> Construction:
        for _, pattern in cls._PATTERNS.items():
            matches = re.match(pattern, string)
            if matches is None:
                continue

            group_dict = matches.groupdict()
            command = group_dict.pop("command")

            if command == "RECT":
                return Rect(
                    int(group_dict["x1"]), int(group_dict["y1"]),
                    int(group_dict["x2"]), int(group_dict["y2"]),
                    group_dict["border"], group_dict["fill"],
                )
            elif command == "CANVAS":
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
