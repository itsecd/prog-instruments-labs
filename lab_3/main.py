import csv
import re

from checksum import calculate_checksum, serialize_result

EMAIL_RE = re.compile(
    r"^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)

HEIGHT_RE = re.compile(r"^[0-2]\.\d{2}$")

INN_RE = re.compile(r"^\d{12}$")

PASSPORT_RE = re.compile(r"^\d{2} \d{2} \d{6}$")

OCCUPATION_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]+(?:[ -][A-Za-zА-Яа-яЁё]+)*$")

LATITUDE_RE = re.compile(r"^-?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d+)?)$")

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

ISSN_RE = re.compile(r"^\d{4}-\d{3}[\dXx]$")

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$")


PATTERNS = {
    "email": EMAIL_RE,
    "height": HEIGHT_RE,
    "inn": INN_RE,
    "passport": PASSPORT_RE,
    "occupation": OCCUPATION_RE,
    "latitude": LATITUDE_RE,
    "hex_color": HEX_COLOR_RE,
    "issn": ISSN_RE,
    "uuid": UUID_RE,
    "time": TIME_RE,
}


def is_valid(column: str, value: str | None) -> bool:

    pattern = PATTERNS.get(column)
    if pattern is None:
        return False
    return pattern.fullmatch(value) is not None


def validate_csv(csv_path: str) -> str:
    invalid_rows = []

    with open(csv_path, newline="", encoding="utf-16") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")

        for row_index, row in enumerate(reader):
            for column, value in row.items():
                if column is None:
                    continue

                column = column.strip()
                if not is_valid(column, value):
                    # print(f"row: {row_index}, err: {column}: {value}")
                    invalid_rows.append(row_index)
                    break

    checksum = calculate_checksum(list(invalid_rows))
    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")
    return checksum


def main():
    VARIANT = 71
    checksum = validate_csv(f"{VARIANT}.csv")
    serialize_result(VARIANT, checksum)


if __name__ == "__main__":
    main()
