from ..models.construction import Construction


class Parser:
    _PATTERNS = {
        "canvas": r"^\# (?P<cmd>CANVAS) +\"(?P<fill>.)\" +\((?P<width>\d+),(?P<height>\d+)\)\s*\#*$",
        "rect": r"^(?P<cmd>RECT) +\"(?P<border>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)( FILL \"(?P<fill>.)\")?\s*\#*$",
    }

    @classmethod
    def _parse_string(cls, string: str) -> Construction:
        pass

    @classmethod
    def parse(cls, text: str) -> list[Construction]:
        pass
