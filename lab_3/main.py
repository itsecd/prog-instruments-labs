"""
Скрипт для валидации CSV-файла 82.csv.

Проверяет 10 полей:
telephone, http_status_message, snils, identifier, ip_v4,
longitude, blood_type, isbn, locale_code, date.

Собирает номера невалидных строк и сохраняет контрольную сумму
в result.json
"""

import csv
import re
from datetime import datetime

import checksum


# 1. Телефон: строгий формат +7-(XXX)-XXX-XX-XX
TELEPHONE_RE = re.compile(
    r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$"
)

# 2. HTTP-статус: код 100-599, пробел и текстовое описание
HTTP_STATUS_MESSAGE_RE = re.compile(
    r"^[1-5]\d{2} [A-Za-z][A-Za-z -]*$"
)

# 3. СНИЛС: ровно 11 цифр
SNILS_RE = re.compile(r"^\d{11}$")

# 4. Идентификатор: 2 цифры - 2 цифры / 2 цифры
IDENTIFIER_RE = re.compile(
    r"^\d{2}-\d{2}/\d{2}$"
)

# 5. IPv4: каждый октет от 0 до 255
IP_V4_RE = re.compile(
    r"^(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\."
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\."
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\."
    r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
)

# 6. Долгота WGS84: от -180 до 180, с точкой и дробной частью
LONGITUDE_RE = re.compile(
    r"^-?(?:180(?:\.0+)?|(?:1[0-7]\d|[0-9]?\d)\.\d+)$"
)

# 7. Группа крови: A, B, AB, O с плюсом или Unicode-минусом
BLOOD_TYPE_RE = re.compile(
    r"^(?:AB|A|B|O)(?:\+|\u2212)$"
)

# 8. ISBN: 10 или 13 цифр, допускаются дефисы между цифрами

ISBN_RE = re.compile(
    r"^(?:\d-\d{5}-\d{3}-\d|\d{3}-\d-\d{5}-\d{3}-\d)$"
)

# 9. locale_code: язык 2-3 строчные буквы, опционально регион через дефис
LOCALE_CODE_RE = re.compile(
    r"^[a-z]{2,3}(?:-[a-z]{2})?$"
)

# 10. Дата: формат YYYY-MM-DD
DATE_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"
)


PATTERNS = {
    "telephone": TELEPHONE_RE,
    "http_status_message": HTTP_STATUS_MESSAGE_RE,
    "snils": SNILS_RE,
    "identifier": IDENTIFIER_RE,
    "ip_v4": IP_V4_RE,
    "longitude": LONGITUDE_RE,
    "blood_type": BLOOD_TYPE_RE,
    "isbn": ISBN_RE,
    "locale_code": LOCALE_CODE_RE,
    "date": DATE_RE,
}


def is_valid_date(value: str) -> bool:
    """
    Проверяет дату на соответствие формату и календарную корректность.
    Например, отсеивает 2023-02-29 и 2021-04-31.
    """
    if not DATE_RE.fullmatch(value):
        return False

    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_row(row: dict) -> bool:
    """
    Проверяет все поля одной строки CSV.
    Возвращает True, если все поля валидны, иначе False.
    """
    for field, pattern in PATTERNS.items():
        value = row.get(field, "")

        if field == "date":
            if not is_valid_date(value):
                return False
        else:
            if not pattern.fullmatch(value):
                return False

    return True


def main() -> None:
    """
    Читает 82.csv, находит невалидные строки,
    считает контрольную сумму и сохраняет result.json.
    """
    csv_path = "82.csv"
    invalid_rows = []

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")


        for line_number, row in enumerate(reader, start=0):
            if not validate_row(row):
                invalid_rows.append(line_number)

    checksum_value = checksum.calculate_checksum(invalid_rows)

    checksum.serialize_result(82, checksum_value)

    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum_value}")


if __name__ == "__main__":
    main()