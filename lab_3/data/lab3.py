import re
import csv
import json
import hashlib
from pathlib import Path

PATTERNS = {
    'telephone':   r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$',
    'height':      r'^[12]\.\d{2}$',
    'snils':       r'^\d{11}$',
    'identifier':  r'^\d{2}-\d{2}/\d{2}$',
    'occupation':  r'^[A-Za-zА-Яа-яёЁ][A-Za-zА-Яа-яёЁ\-]*(?: [A-Za-zА-Яа-яёЁ][A-Za-zА-Яа-яёЁ\-]*)*$',
    'blood_type':  r'^(A|B|AB|O)[+\u2212]$',
    'issn':        r'^\d{4}-\d{4}$',
    'locale_code': r'^[a-z]{2}(?:-[a-z]{2})?$',
    'date':        r'^\d{4}-\d{2}-\d{2}$',
}

LONGITUDE_FMT = re.compile(r'^-?\d{1,3}\.\d+$')


def validate_longitude(value: str) -> bool:
    if not LONGITUDE_FMT.fullmatch(value):
        return False
    try:
        x = float(value)
    except ValueError:
        return False
    return -180.0 <= x <= 180.0


def validate_field(value: str, field_name: str) -> bool:
    if field_name == 'longitude':
        return validate_longitude(value)
    pattern = PATTERNS.get(field_name)
    if not pattern:
        return True
    return bool(re.fullmatch(pattern, value))


def validate_csv(file_path: Path):
    invalid_rows = []
    field_fail = {f: [] for f in list(PATTERNS) + ['longitude']}

    with open(file_path, 'r', encoding='utf-16') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row_num, row in enumerate(reader, start=0):
            row_bad = False
            for field_name, value in row.items():
                if field_name in field_fail and not validate_field(value, field_name):
                    field_fail[field_name].append((row_num, value))
                    row_bad = True
            if row_bad:
                invalid_rows.append(row_num)

    return invalid_rows, field_fail


def calculate_checksum(invalid_rows: list[int]) -> str:
    invalid_rows.sort()
    return hashlib.md5(json.dumps(invalid_rows).encode('utf-8')).hexdigest()


def main():
    csv_file = Path('84.csv')
    variant = 84

    invalid_rows, field_fail = validate_csv(csv_file)
    checksum = calculate_checksum(invalid_rows)

    print(f"Найдено {len(invalid_rows)} невалидных строк")
    print(f"Контрольная сумма (MD5): {checksum}")
    print("\n=== Статистика по полям (ожидаем ~100 на поле) ===")
    for field, lst in field_fail.items():
        print(f"{field}: {len(lst)}")
        for rn, val in lst[:6]:
            print(f"    row {rn}: {val!r}")

    result = {'variant': variant, 'checksum': checksum, 'invalid_rows': invalid_rows}
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nРезультат сохранён в result.json")


if __name__ == "__main__":
    main()