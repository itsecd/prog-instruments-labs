import os


class Iterator:
    def __init__(self, path: str) -> None:
        """
        Constructor of iterator
        """
        self.data = path
        self.folder_element = os.listdir(self.data)
        self.limit = len(self.folder_element)
        self.counter = 0

    def __next__(self) -> str:
        """
        This method return absolute path of next element from dataset
        """
        if self.counter < self.limit:
            absolute_path = os.path.join(self.data, self.folder_element[self.counter]).replace("\\", "/")
            self.counter += 1
            return absolute_path
        else:
            raise StopIteration
        
    def previous(self) -> str:
        """
        This method return absolute path of previous element from dataset
        """
        if self.counter > 0:
            self.counter -= 1
            absolute_path = os.path.join(self.data, self.folder_element[self.counter]).replace("\\", "/")
            return absolute_path
        else:
            raise StopIteration