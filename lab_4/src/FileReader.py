import json
import Entry


class FileReader:
    """Reads file data by path name"""
    data: list[Entry]

    def __init__(self, path) -> None:
        """Contstructor: writes data to class self"""
        self.data = json.load(open(path, encoding='windows-1251'))

    def getData(self) -> list[Entry]:
        """Returns all nodes"""
        return self.data
