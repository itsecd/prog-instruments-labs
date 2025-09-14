import re

_COMMENT = r"\#"  # "#"
_START = r"^"
_END = fr"\s*{_COMMENT}*$"  # "... #..."

_RAW_NUM = r"(?P<{}>\d+)"  # "123"



if __name__ == "__main__":
    pass