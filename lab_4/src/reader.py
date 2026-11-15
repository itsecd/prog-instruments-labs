"""Чтение CSV-файла с данными лабораторной."""
import csv
from collections.abc import Iterator

from src.config import CSV_DELIMITER, CSV_ENCODING


def read_csv(filepath) -> Iterator[dict]:
    """Открывает CSV и возвращает итератор по строкам-словарям.

    :param filepath: путь к CSV-файлу
    :yield: словарь {имя_колонки: значение}
    """
    with open(filepath, encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
        yield from reader
