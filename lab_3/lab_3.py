import csv
import re
import json
from checksum import calculate_checksum

CSV_FILE_PATH = "1.csv"
VARIANT_NUMBER = 1

REGEX_PATTERNS = {
    "email": r"^[a-zA-Z0-9][a-zA-Z0-9._-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "http_status_message": r"^\d{3}\s[A-Z][a-zA-Z\s]*$",
    "snils": r"^\d{11}$",
    "passport": r"^\d{2}\s\d{2}\s\d{6}$",
    "ip_v4": r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$",
    "longitude": r"^-?(\d{1,2}|1[0-7]\d|180)(\.\d+)?$",
    "hex_color": r"^#[0-9a-fA-F]{6}$",
    "isbn": r"^(\d+[-]){3,4}[\dX]$",
    "locale_code": r"^[a-z]{2,3}(-[a-z]{2,4})?$",
    "time": r"^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)(\.\d{1,6})?$"
}


def validate_row(row_data: dict) -> bool:
    """
    Проверяет соответствие полей и их регулярных выражений.
    """
    for key, value in row_data.items():
        if key in REGEX_PATTERNS and not re.match(REGEX_PATTERNS[key], value):
            return False
    return True


def main():
    """
    Основная функция: читает CSV-файл, валидирует строки и считает контрольную сумму,
    которую записывает в виде результата в JSON-файл.
    """
    invalid_rows_indices = []
    try:
        with open(CSV_FILE_PATH, "r", encoding="utf-16") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=";")
            for i, row in enumerate(reader):
                if not validate_row(row):
                    invalid_rows_indices.append(i)

    except FileNotFoundError:
        print(f"Ошибка: Файл '{CSV_FILE_PATH}' не найден.")
        return
    except Exception as e:
        print(f"Произошла ошибка при чтении или обработке файла: {e}")
        return

    checksum = calculate_checksum(invalid_rows_indices)

    print(f"Найдено невалидных строк: {len(invalid_rows_indices)}")
    print(f"Контрольная сумма: {checksum}")

    result_data = {
        "variant": VARIANT_NUMBER,
        "checksum": checksum
    }
    with open("result.json", "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    print(f"Результат успешно записан в файл 'result.json'")


if __name__ == "__main__":
    main()