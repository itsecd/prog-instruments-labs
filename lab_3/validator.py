import csv
import re
from re import Pattern
from typing import List, Dict
from checksum import calculate_checksum, serialize_result

FILENAME = 'lab_3/31.csv'


REGEX_FORMATS: Dict[ str, Pattern[str] ] = {
    'email': re.compile(r'[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*@[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+'),
    'height': re.compile(r'[12]\.\d{2}'),
    'inn': re.compile(r'\d{12}'),
    'passport': re.compile(r'(?:\d\d ){2}\d{6}'),
    'occupation': re.compile(r'[ a-zA-Zа-яА-ЯёЁ-]+'),
    'latitude': re.compile(r'\-?(?:((?:1[0-7]\d|\d{1,2})\.[0-9]{1,6})|180\.0{1,6})'),
    'hex_color': re.compile(r'#[0-9a-f]{6}'),
    'issn': re.compile(r'\d{4}\-\d{4}'),
    'uuid': re.compile(r'[0-9a-f]{8}\-(?:[0-9a-f]{4}\-){3}[0-9a-f]{12}'),
    'time': re.compile(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{6}'),
}


def read_csv(filename: str) -> List[ Dict[str, str] ]:
    """
    Читает данные из .csv файла.

    Args:
        filename: Название или путь до .csv файла.

    Returns:
        List[ Dict[ str, str ] ]: Массив со всеми строками из .csv файла, оформленные в виде словарей.
    """

    with open( filename, 'r', encoding = 'utf-16' ) as csv_file:
        reader = csv.DictReader( csv_file, delimiter = ';' )
        return list( reader )


def check_regex_match( row: Dict[str, str] ) -> bool:
    """
    Проверяет подходит ли полностью строка (словарь) под установленные регулярные выражения. 

    Args:
        row: Словарь с названиями столбцов и соответствующими значениями.

    Returns:
        bool: Если хотя бы одно из полей не проходит проверку, то возвращается False. В ином случае - True.
    """

    for column_name, value in row.items():
        if re.fullmatch( REGEX_FORMATS[column_name], value ) is None:
            return False
        
    return True


def collect_failed_regex_rows(rows: List[ Dict[str, str] ]) -> List[int]:
    """
    Создает список из индексов невалидных строк (словарей). 

    Args:
        rows: Все строки (словари).

    Returns:
        List[int]: Список из индексов невалидных строк.
    """

    result = []

    for i, row in enumerate(rows):
        if check_regex_match(row) is False:
            result.append(i)

    return result


if __name__ == "__main__":
    rows = read_csv(FILENAME)
    invalid_indexes = collect_failed_regex_rows(rows)
    cash = calculate_checksum(invalid_indexes)
    serialize_result(31, cash)
    