import json

from Entry import Entry


class FileReader:
    """Reads file data by path name"""
    data: list[Entry]

    def __init__(self, path: str) -> None:
        """Contstructor: writes data to class self"""
        with open(path, encoding='windows-1251') as f:
            data = json.load(f)

        self.data = data

    def get_data(self) -> list[Entry]:
        """Returns all nodes"""
        return self.data
