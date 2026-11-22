import argparse
import csv
import json
import re
from pathlib import Path

from checksum import calculate_checksum


VARIANT = 53
OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
HEX = r"[0-9a-fA-F]"

PATTERNS = {
    "email": re.compile(
        r"[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*@(?:[a-zA-Z0-9]+\.)+[a-zA-Z]{2,}"
    ),
    "http_status_message": re.compile(r"[1-5][0-9]{2} [A-Za-z]+(?: [A-Za-z]+)*"),
    "inn": re.compile(r"[0-9]{12}"),
    "passport": re.compile(r"[0-9]{2} [0-9]{2} [0-9]{6}"),
    "ip_v4": re.compile(rf"{OCTET}(?:\.{OCTET}){{3}}"),
    "latitude": re.compile(r"-?(?:[0-8]?[0-9](?:\.[0-9]+)?|90(?:\.0+)?)"),
    "hex_color": re.compile(rf"#{HEX}{{6}}"),
    "isbn": re.compile(r"(?:[0-9]{3}-)?[0-9]-[0-9]{5}-[0-9]{3}-[0-9]"),
    "uuid": re.compile(rf"{HEX}{{8}}-(?:{HEX}{{4}}-){{3}}{HEX}{{12}}"),
    "time": re.compile(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{1,6}"),
}


def is_valid(field: str, value: str) -> bool:
    """Проверяет значение целиком по правилу для его столбца."""
    return PATTERNS[field].fullmatch(value) is not None


def validate_csv(file_path: Path) -> list[int]:
    """Возвращает номера ошибочных строк. Первая строка данных имеет номер 0."""
    invalid_rows = []
    with file_path.open(encoding="utf-16", newline="") as source:
        reader = csv.DictReader(source, delimiter=";")
        if reader.fieldnames != list(PATTERNS):
            raise ValueError("Заголовки или порядок столбцов не соответствуют варианту 53")

        for number, row in enumerate(reader):
            if None in row or any(
                not is_valid(field, row[field] or "") for field in PATTERNS
            ):
                invalid_rows.append(number)
    return invalid_rows


def save_result(variant: int, checksum: str) -> None:
    """Сохраняет результат рядом с main.py в формате для проверки."""
    result = {"variant": str(variant), "checksum": checksum}
    path = Path(__file__).with_name("result.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка CSV для варианта 53")
    parser.add_argument("csv_file", type=Path, help="Путь к исходному CSV-файлу")
    args = parser.parse_args()

    invalid_rows = validate_csv(args.csv_file)
    checksum = calculate_checksum(invalid_rows)
    save_result(VARIANT, checksum)

    print(f"Всего невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")
    print("Результат сохранён в result.json")


if __name__ == "__main__":
    main()
