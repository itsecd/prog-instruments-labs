import csv
import re

from checksum import calculate_checksum, serialize_result

VARIANT = 7
CSV_PATH = "7.csv"

PATTERNS = {
    "email": re.compile(
        r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*"
        r"@(?:[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\.)+[A-Za-z]{2,}$"
    ),
    "height": re.compile(r"^(?:1\.\d{2}|2\.00)$"),
    "inn": re.compile(r"^\d{12}$"),
    "passport": re.compile(r"^\d{2} \d{2} \d{6}$"),
    "occupation": re.compile(
        r"^[A-Za-zА-Яа-яЁё]+(?:[ \-][A-Za-zА-Яа-яЁё]+)*$"
    ),
    "latitude": re.compile(
        r"^-?(?:90(?:\.0+)?|(?:0|[1-8]?\d)(?:\.\d+)?)$"
    ),
    "hex_color": re.compile(r"^#[0-9a-fA-F]{6}$"),
    "issn": re.compile(r"^\d{4}-\d{4}$"),
    "uuid": re.compile(
        r"^[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}$"
    ),
    "time": re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}$"),
}


def is_valid_row(row: dict) -> bool:
    """
    Проверяет одну строку CSV-файла на соответствие всем регулярным выражениям
    :param row: словарь вида {имя_столбца: значение} — одна строка, прочитанная через csv.DictReader
    :return: True, если все поля валидны, иначе False
    """
    return all(
        PATTERNS[column].fullmatch(row.get(column) or "")
        for column in PATTERNS
    )

def serach_invalid_rows(CSV_PATH: str) -> list[int]:
    """
    Находит номера невалидных строк CSV
    :param CSV_PATH: путь к csv файлу
    :return: возвращает список номеров невалидных строк
    """
    invalid_rows = []

    with open(CSV_PATH, newline="", encoding="utf-16") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")

        for row_number, row in enumerate(reader):
            if not is_valid_row(row):
                invalid_rows.append(row_number)

    return invalid_rows


def main() -> None:
    invalid_rows = serach_invalid_rows(CSV_PATH)

    checksum = calculate_checksum(invalid_rows)
    serialize_result(VARIANT, checksum)

    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()