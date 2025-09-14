import re

_COMMENT = r"\#"  # "#"
_START = r"^"
_END = fr"\s*{_COMMENT}*$"  # "... #..."

_RAW_NUM = r"(?P<{}>\d+)"  # "123"
_RAW_SYMB = r"\"(?P<{}>.)\""  # '"s"'

_RAW_COORDS_2D = fr"\({_RAW_NUM},\s*{_RAW_NUM}\)"  # "(1, 2)"
_RAW_COORDS_3D = fr"\({_RAW_NUM},\s*{_RAW_NUM},\s*{_RAW_NUM}\)"  # "(1, 2, 3)"
_RAW_FROM_TO_2D = fr"{_RAW_COORDS_2D}\s+->\s+{_RAW_COORDS_2D}"  # "(1,2)  -> (3, 4)"
_RAW_FILL_ARG = fr"FILL\s+{_RAW_SYMB}"  # 'FILL  "X"'


if __name__ == "__main__":
    pass