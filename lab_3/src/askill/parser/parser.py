import re
from ..models import Construction, Rect, Canvas


class Parser:
    _PATTERNS = {
        "canvas": r"^\# (?P<command>CANVAS) +\"(?P<fill>.)\" +\((?P<width>\d+),(?P<height>\d+)\)\s*\#*$",
        "rect": r"^(?P<command>RECT) +\"(?P<border>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)( FILL \"(?P<fill>.)\")?\s*\#*$",
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
        pass

