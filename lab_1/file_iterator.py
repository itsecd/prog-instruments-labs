from collections.abc import Iterator


class FileIterator:
    """
    Итератор, читающий файл построчно.
    """

    def __init__(self, file_path: str) -> None:
        """
        Инициализирует итератор и читает файл целиком.
        :param file_path: Путь к файлу
        """
        self.filepath = file_path
        try:
            with open(file_path, encoding="utf-8") as file:
                self._lines: list[str] = file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден") from None
        self._index = 0

    def __iter__(self) -> Iterator[str]:
        """
        Возвращает итератор строк.
        """
        return self

    def __next__(self) -> str:
        """
        Возвращает следующую строку из файла.
        """
        if self._index >= len(self._lines):
            raise StopIteration
        line = self._lines[self._index]
        self._index += 1
        return line
