COMMENT = r"\#"  # "#"
START = r"^\s*"
END = fr"\s*{COMMENT}?.*$"  # "... #..."

RAW_NUM = r"(?P<{}>\d+)"  # "123"
RAW_SYMB = r"\"(?P<{}>.)\""  # '"s"'

RAW_COORDS_2D = fr"\({RAW_NUM},\s*{RAW_NUM}\)"  # "(1, 2)"
RAW_COORDS_3D = fr"\({RAW_NUM},\s*{RAW_NUM},\s*{RAW_NUM}\)"  # "(1, 2, 3)"
RAW_FROM_TO_2D = fr"{RAW_COORDS_2D}\s+->\s+{RAW_COORDS_2D}"  # "(1,2)  -> (3, 4)"
RAW_FILL_ARG = fr"FILL\s+{RAW_SYMB}"  # 'FILL  "X"'

CANVAS_CMD = fr"{START}{COMMENT}\s*CANVAS\s+{RAW_SYMB.format('fill')}\s+{RAW_COORDS_2D.format('width', 'height')}{END}"
"""Examples:
```
'# CANVAS "x" (24, 24)'
```
"""

RECT_CMD = fr"{START}RECT\s+{RAW_SYMB.format('border')}\s+{RAW_FROM_TO_2D.format('x1', 'y1', 'x2', 'y2')}( {RAW_FILL_ARG.format('fill')})?{END}"
"""Examples:
```
'RECT "X" (1,2) -> (3, 4) FILL "X" # comment'
'RECT "Y" (0,0) -> (1, 1)'
```
"""

CIRCLE_CMD = fr"{START}CIRCLE\s+{RAW_SYMB.format('border')}\s+{RAW_COORDS_3D.format('x', 'y', 'r')}( {RAW_FILL_ARG.format('fill')})?{END}"
"""Examples:
```
'CIRCLE "A" (1,2, 4) FILL "2"   # ASD'
```
"""

LINE_CMD = fr"{START}LINE\s+{RAW_SYMB.format('fill')}\s+{RAW_FROM_TO_2D.format('x1', 'y1', 'x2', 'y2')}{END}"
"""Examples:
```
'LINE "U" (0,0) -> (1, 1)# ASD'
```
"""

PATTERNS = {
    "canvas": CANVAS_CMD,
    "rect": RECT_CMD,
    "line": LINE_CMD,
    "circle": CIRCLE_CMD,
}
