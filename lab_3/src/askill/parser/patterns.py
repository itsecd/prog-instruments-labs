_PATTERNS = {
    "canvas": r"^\# CANVAS +\"(?P<fill>.)\" +\((?P<width>\d+),(?P<height>\d+)\)\s*\#*$",
    "rect": r"^RECT +\"(?P<border>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)( FILL \"(?P<fill>.)\")?\s*\#*$",
    "circle": r"^CIRCLE +\"(?P<border>.)\" +\((?P<x>\d+),(?P<y>\d+),(?P<r>\d+)\)( FILL \"(?P<fill>.)\")?\s*$",
    "line": r"^LINE +\"(?P<fill>.)\" +\((?P<x1>\d+),(?P<y1>\d+)\) -> \((?P<x2>\d+),(?P<y2>\d+)\)\s*$",
}