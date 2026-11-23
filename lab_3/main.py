import csv
import json 
import re
from typing import Dict, List
from checksum import calculate_checksum,    serialize_result


PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(
        r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+$"
    ),
    "http_status_message": re.compile(r"^\d{3}\s+[A-Za-z\s]+$"),
    "snils": re.compile(r"^\d{11}$"),
    "passport": re.compile(r"^\d{2}\s+\d{2}\s+\d{6}$"),
    "ip_v4": re.compile(
        r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
    ),
    "longitude": re.compile(
        r"^[-+]?(?:180(?:\.0+)?|(?:1[0-7]\d|\d{1,2})(?:\.\d+)?)$"
    ),
    "hex_color": re.compile(r"^#[0-9a-fA-F]{6}$"),
    "isbn": re.compile(r"^(?:\d{3}-)?(?:\d+-){3}[\dX]$"),
    "locale_code": re.compile(r"^[a-z]{2}(?:-[a-z]{2,})?$"),
    "time": re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$"),
}
VARIANT_NUMBER = 25
FILE_PATH_CSV = f"25.csv"


def check_row_valid(row: Dict[str, str]) -> bool:
    for col_name, pattern in PATTERNS.items():
        val = row.get(col_name, " ")
        if not pattern.match(val):
            return False
    return True         


def find_invalid_row(csv_path: str) -> List[int]:
    invalid_indexs: List[int] = []

    with open(csv_path, "r", encoding="utf-16") as file:
        reader = csv.DictReader(file, delimiter=";")
        for idx, row in enumerate(reader):
            if not check_row_valid(row):
                invalid_indexs.append(idx)

    return invalid_indexs                


def main() -> None:
    invalid_indexs = find_invalid_row(FILE_PATH_CSV)
  
    if len(invalid_indexs) != 1000:
        print(f"Warning: Expected exactly 1000 invalid rows, but found {len(invalid_indexs)}." 
              "Please check your regular expressions!")        
    checksum = calculate_checksum(invalid_indexs)
    print(f"Calculated Checksum: {checksum}")

    serialize_result(VARIANT_NUMBER, checksum);

    print("Update successfully!")


if __name__ == "__main__":
    main()
        