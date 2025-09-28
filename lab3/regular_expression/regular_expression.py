import re


def check_validate(string: str, regexp: str) -> bool:
    return bool(re.fullmatch(rf"{regexp}", string))