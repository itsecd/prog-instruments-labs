import csv
import re
import json
from checksum import calculate_checksum

# --- Константы ---
CSV_FILE_PATH = "10.csv"
VARIANT_NUMBER = 10

# --- Словарь с регулярными выражениями для каждого поля ---
REGEX_PATTERNS = {
    "telephone": r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$",
    "http_status_message": r"^\d{3}\s[A-Z][a-zA-Z\s]*$",
    "snils": r"^\d{11}$",
    "identifier": r"^\d{2}-\d{2}\/\d{2}$",
    "ip_v4": r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$",
    "longitude": r"^-?(\d{1,2}|1[0-7]\d|180)(\.\d+)?$",
    "blood_type": r"^(A|B|AB|O)[\+\-\u2212]$",
    "isbn": r"^(\d+[-]){3,4}[\dX]$",
    "locale_code": r"^[a-z]{2,3}(-[a-z]{2})?$",
    "date": r"^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
}


def validate_row(row_data: dict) -> bool:
    """
    Проверяет, что все поля в строке соответствуют своим регулярным выражениям.
    """
    for key, value in row_data.items():
        if not re.match(REGEX_PATTERNS[key], value):
            return False
    return True


def main():
    """
    Основная функция: читает CSV, валидирует строки, считает контрольную сумму
    и записывает результат в JSON.
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