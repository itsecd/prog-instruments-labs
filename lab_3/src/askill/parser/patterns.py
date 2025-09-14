import re

COMMENT = r"\#"  # "#"
START = r"^"
END = fr"\s*{COMMENT}*$"  # "... #..."

RAW_NUM = r"(?P<{}>\d+)"  # "123"
RAW_SYMB = r"\"(?P<{}>.)\""  # '"s"'

RAW_COORDS_2D = fr"\({RAW_NUM},\s*{RAW_NUM}\)"  # "(1, 2)"
RAW_COORDS_3D = fr"\({RAW_NUM},\s*{RAW_NUM},\s*{RAW_NUM}\)"  # "(1, 2, 3)"
RAW_FROM_TO_2D = fr"{RAW_COORDS_2D}\s+->\s+{RAW_COORDS_2D}"  # "(1,2)  -> (3, 4)"
RAW_FILL_ARG = fr"FILL\s+{RAW_SYMB}"  # 'FILL  "X"'

CANVAS_CMD = fr"{START}{COMMENT}\s*CANVAS\s+{RAW_SYMB.format('fill')}\s+{RAW_COORDS_2D.format('width', 'height')}{END}"
# '# CANVAS "x" (24, 24)'


if __name__ == "__main__":
    m = re.match(CANVAS_CMD, '# CANVAS "x" (24, 24)')

    print(m.groupdict())