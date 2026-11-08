import csv
import re

from checksum import calculate_checksum, serialize_result

CSV_PATH = "97.csv"
RESULT_PATH = "result.json"
VARIANT = 97


PATTERNS = {
    "email": r"^[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*@[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)+$",
    "http_status_message": r"^[1-5]\d\d [A-Za-z][A-Za-z ]*$",
    "snils": r"^\d{11}$",
    "passport": r"^\d{2} \d{2} \d{6}$",
    "ip_v4": (r"^((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
              r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"),
    "longitude": (r"^-?(180(\.\d{1,6})?|1[0-7]\d(\.\d{1,6})?"
                  r"|\d{1,2}(\.\d{1,6})?)$"),
    "hex_color": r"^#[0-9a-f]{6}$",
    "isbn": r"^(\d{3}-\d-\d{5}-\d{3}-\d|\d-\d{5}-\d{3}-\d)$",
    "locale_code": r"^[a-z]{2}(-[a-z]{2})?$",
    "time": r"^([01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$",
}


def read_csv(path):
    """Читает CSV-файл в кодировке UTF-16 и возвращает (заголовок, строки)."""
    with open(path, encoding="utf-16") as file:
        reader = csv.reader(file, delimiter=";")
        rows = list(reader)
    return rows[0], rows[1:]


def find_invalid_rows(header, rows):
    """Возвращает номера строк, содержащих хотя бы одно невалидное поле."""
    compiled = {name: re.compile(PATTERNS[name]) for name in header}
    invalid = set()
    for index, row in enumerate(rows):
        for column, name in enumerate(header):
            if not compiled[name].fullmatch(row[column]):
                invalid.add(index)
                break
    return sorted(invalid)


if __name__ == "__main__":
    header, rows = read_csv(CSV_PATH)
    invalid_rows = find_invalid_rows(header, rows)
    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)
