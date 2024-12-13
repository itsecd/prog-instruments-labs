import json
import hashlib
import csv
import re
from typing import Dict, List

patterns: Dict[str, str] = {
    "email": r"^[a-z0-9]+(?:[._][a-z0-9]+)*\@[a-z]+(?:\.[a-z]+)+$",
    "height": r"^[1-2]\.\d{2}$",
    "inn": r"^\d{12}$",
    "passport": r"^\d{2}\s\d{2}\s\d{6}$",
    "occupation": r"[a-zA-Zа-яА-ЯёЁ -]+",
    "latitude": r"^-?(90|[0-8]?[0-9])\.\d+$",
    "hex_color": r"^\#[0-9a-fA-F]{6}$",
    "issn": r"^\d{4}-\d{4}$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "time": r"^(2[0-3]|[0-1][0-9]):[0-5][0-9]:[0-5][0-9]\.\d{6}$"
}


def validate_row(fields: List[str]) -> bool:
    """
    Validates each field of the string according to the specified regular expressions.

    :param fields: A list of the values of the string fields.
    :return: False if a mismatched record is found, otherwise True.
    """
    for idx, field in enumerate(fields):
        pattern_key = list(patterns.keys())[idx]
        regex = patterns.get(pattern_key)
        if not re.fullmatch(regex, field):
            return False
    return True


def process_csv(file_path: str) -> List[int]:
    """
    Processes a CSV file for error rows.

    :param file_path: Path to the CSV file.
    :return: List of row lines with errors.
    """
    invalid_rows: List[int] = []
    with open(file_path, newline='', encoding='utf-16') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        next(reader)
        
        for row_number, row in enumerate(reader, start=1):
            if not validate_row(list(row.values())):
                invalid_rows.append(row_number)

    return invalid_rows


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Calculates a checksum for a list of row numbers.

    :param row_numbers: List of row numbers.
    :return: Checksum in MD5 format.
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_to_json(variant: int, checksum: str) -> None:
    """
    Serializes the result to a JSON file.

    :param variant: Variant number.
    :param checksum: Checksum.
    """
    with open("lab_3/path_csv.json", "r", encoding='utf-8') as options_file:
        options = json.load(options_file)
        result_file_path = options["result_file_path"]

    with open(result_file_path, 'w', encoding='utf-8') as file:
        result = {
            "variant": variant,
            "checksum": checksum
        }
        json.dump(result, file, ensure_ascii=False)


if __name__ == "__main__":

    with open("lab_3/path_csv.json", "r", encoding='utf-8') as options_file:
        options = json.load(options_file)

    invalid_data = process_csv(options["csv_file_path"])
    print("Число невалидных строк:", len(invalid_data))

    variant = 47
    checksum = calculate_checksum(invalid_data)
    serialize_to_json(variant, checksum)
