import csv
import logging

from typing import Any


logger = logging.getLogger(__name__)


class MyCsv:
    def __init__(self, csv_path: str, names: bool,  delimiter: str = ";") -> None:
        """
        Читает csv файл по пути csv_path. Заполняет значениями поля классов, такие как:
        self.csv_path - путь до файла csv
        self.names - названия столбцов
        self.data - список не пустых строк из csv файла
        """
        if len(delimiter) > 1:
            logger.error(f"delimiter must consist of 1 character")
            raise ValueError("delimiter must consist of 1 character")
        self.csv_path = csv_path
        with open(csv_path, "r", encoding="utf-16") as csv_reader:
            logger.info(f"Successfully open file '%s'", csv_reader)
            reader = csv.reader(csv_reader, delimiter=delimiter)
            self.data = []
            self.names = []
            for row in reader:
                if names:
                    self.names = row
                    names = False
                    continue
                if len(row) != 0:
                    self.data.append(row)
            logger.info(f"End work with file '%s'", csv_reader)

    def get_values_from_col(self, col_number: int) -> list[Any]:
        """
        Возвращает список значений определённого столбца col_number
        """
        res = []
        for row in self.data:
            res.append(row[col_number])
        return res