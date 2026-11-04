import argparse
import re
import calendar

DATE_PATTERN = re.compile(r"Дата рождения\s*:\s*(\d{1,2}[./-]\d{1,2}[./-]\d{4})")

DATE_VALIDATION_PATTERN = re.compile(
    r"\d{1,2}([./-])\d{1,2}\1\d{4}"
)

MIN_YEAR = 1900
MAX_YEAR = 2025

def date_check(date: str) -> bool:
    """Проверка корректности даты (с учетом високосных годов)"""
    if not DATE_VALIDATION_PATTERN.fullmatch(date):
        return False

    day, month, year = map(int, re.split(r"[./-]", date))

    if not MIN_YEAR <= year <= MAX_YEAR:
        return False

    if not 1 <= month <= 12:
        return False

    max_day = calendar.monthrange(year, month)[1]

    return 1 <= day <= max_day

def read_file(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def parse_records(content: str):
    raw_records = content.strip().split("\n\n")
    return [rec.split("\n") for rec in raw_records]

def extract_dob(record: list[str]) -> str | None:
    """Ищет строку с датой рождения через regex"""
    for line in record:
        match = DATE_PATTERN.match(line)
        if match:
            return match.group(1)
    return None

def validate_records(records):
    invalid_records, valid_records = [], []
    for rec in records:
        dob = extract_dob(rec)
        if dob and date_check(dob):
            valid_records.append(rec)
        else:
            invalid_records.append(rec)
    return invalid_records, valid_records

def print_invalid_records(invalid_records):
    """Выводит анкеты с некорректной датой рождения."""
    print("Анкеты с некорректной датой рождения")

    for record in invalid_records:
        print("\n".join(record))

def save_valid_records(valid_records, output_path):
    """Сохраняет корректные анкеты в файл."""
    with open(output_path, "w", encoding="utf-8") as file:
        for record in valid_records:
            file.write("\n".join(record) + "\n\n")

def print_summary(invalid_count, output_path):
    """Выводит количество некорректных анкет и путь к результату."""
    print(f"\nОбщее количество некорректных анкет {invalid_count}")
    print(f"Результат сохранён в файл {output_path}")

def print_file(invalid_records, valid_records):
    """Организует вывод и сохранение результатов проверки."""
    output_path = "output.txt"

    print_invalid_records(invalid_records)
    save_valid_records(valid_records, output_path)
    print_summary(len(invalid_records), output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Путь до файла с данными")
    args = parser.parse_args()
    content = read_file(args.filename)
    records = parse_records(content)
    invalid_records, valid_records = validate_records(records)
    print_file(invalid_records, valid_records)


if __name__ == "__main__":
    main()