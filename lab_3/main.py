import csv
import re
from pathlib import Path

from checksum import calculate_checksum, serialize_result


OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
PATTERNS = {
    "telephone": r"\+7-\([0-9]{3}\)-[0-9]{3}-[0-9]{2}-[0-9]{2}",
    "http_status_message": r"[0-9]{3} [A-Za-z]+(?: [A-Za-z]+)*",
    "inn": r"[0-9]{12}",
    "identifier": r"[0-9]{2}-[0-9]{2}/[0-9]{2}",
    "ip_v4": rf"{OCTET}(?:\.{OCTET}){{3}}",
    "latitude": r"-?(?:[0-9]|[1-8][0-9])(?:\.[0-9]+)?|"
                r"-?90(?:\.0+)?",
    "blood_type": r"(?:AB|A|B|O)[+\u2212]",
    "isbn": r"(?:[0-9]{3}-)?[0-9]-[0-9]{5}-[0-9]{3}-[0-9]",
    "uuid": r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
    "date": r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])",
}


def find_invalid_rows(csv_path):
    """Вернуть номера неверных строк и число ошибок в каждом столбце."""
    patterns = {
        name: re.compile(pattern) for name, pattern in PATTERNS.items()
    }
    invalid_rows = []
    errors = dict.fromkeys(patterns, 0)

    with csv_path.open(encoding="utf-16", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        if reader.fieldnames != list(patterns):
            raise ValueError("Unexpected CSV columns")

        for row_number, row in enumerate(reader):
            if None in row or None in row.values():
                raise ValueError(f"Wrong number of cells in row {row_number}")
            valid = True
            for field, pattern in patterns.items():
                if pattern.fullmatch(row[field]) is None:
                    errors[field] += 1
                    valid = False
            if not valid:
                invalid_rows.append(row_number)

    return invalid_rows, errors


def main():
    """Проверить файл и сохранить результат в result.json."""
    csv_path = Path(__file__).with_name("62.csv")
    invalid_rows, errors = find_invalid_rows(csv_path)
    checksum = calculate_checksum(invalid_rows)
    serialize_result(62, checksum)

    for field, count in errors.items():
        print(f"{field}: {count}")
    print(f"Invalid rows: {len(invalid_rows)}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()