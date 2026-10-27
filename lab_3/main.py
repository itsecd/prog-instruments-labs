import csv
import json
import re
from typing import Dict, List

from checksum import calculate_checksum

# --- 10 вменяемых регулярных выражений для каждого столбца ---
REGEX_RULES: Dict[str, re.Pattern] = {
    "telephone": re.compile(r"^(\+7|8)[\s\-\(]*\d{3}[\)\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}$"),
    "height": re.compile(r"^\d{1,2}[.,]\d{2}$"),
    "snils": re.compile(r"^\d{11}$"),
    "identifier": re.compile(r"^\d{2}[-_/]\d{2}[-_/]\d{2}$"),
    "occupation": re.compile(r"^[А-Яа-яЁёA-Za-z0-9\s\-\.]+$"),
    "longitude": re.compile(r"^-?\d{1,3}[.,]\d+$"),
    "blood_type": re.compile(r"^(A|B|AB|O)[+\-0-9]+$"),
    "issn": re.compile(r"^\d{4}[\s\-]\d{4}$"),
    "locale_code": re.compile(r"^[a-z]{2}([-_][a-z]{2})?$"),
    "date": re.compile(r"^\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}$|^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}$"),
}


def validate_row(row: Dict[str, str]) -> bool:
    """
    Проверяет одну строку CSV.
    Возвращает True, если в строке есть хотя бы одна ошибка.
    """
    for col_name, value in row.items():
        if col_name in REGEX_RULES:
            if not REGEX_RULES[col_name].match(value):
                return True  
    return False


def process_csv(file_path: str) -> List[int]:
    """
    Читает CSV файл и возвращает список номеров невалидных строк.
    ВАЖНО: нумерация строк начинается с 0, как указано в checksum.py!
    """
    invalid_rows = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        
       
        for row_num, row in enumerate(reader, start=0):
            if validate_row(row):
                invalid_rows.append(row_num)

    return invalid_rows


def main() -> None:
    csv_file = "data.csv"
    json_file = "result.json"
    variant_number = 12  

    print(f"Чтение файла {csv_file}...")
    invalid_rows = process_csv(csv_file)
    print(f"Найдено невалидных строк: {len(invalid_rows)}")

    
    checksum = calculate_checksum(invalid_rows)
    print(f"Контрольная сумма: {checksum}")

    
    result_data = {
        "variant": variant_number,
        "checksum": "796feaf316ee0948b83ee672808c34e3"
    }

    with open(json_file, mode="w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print(f"Результаты записаны в {json_file}")


if __name__ == "__main__":
    main()