from enum import Enum

DEFAULT_DIRECTORY = "C:/Users/ct/PycharmProjects/isb/lab3"
FILTER = "JSON Files (*.json)"


class IconTypes(Enum):
    Critical = ("critical",)
    Warning = ("warning",)
    Question = ("question",)
    Information = ("information",)
    NoIcon = ""
