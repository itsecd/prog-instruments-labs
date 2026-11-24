import csv
import re
from checksum import calculate_checksum, serialize_result

def main():
    variant = 12
    
    # Ваши регулярные выражения
    regex_patterns = {
        "telephone": r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$",
        "height": r"^[12]\.\d{2}$",
        "snils": r"^\d{11}$",
        "identifier": r"^\d{2}-\d{2}/\d{2}$",
        "occupation": r"^[A-Za-zА-Яа-яЁё\s\-]+$",
        "longitude": r"^-?\d+\.\d+$",
        "blood_type": r"^(A|B|AB|O)[+\-−]$",
        "issn": r"^\d{4}-\d{4}$",
        "locale_code": r"^[a-z]{2}(-[a-z]{2})?$",
        "date": r"^\d{4}-\d{2}-\d{2}$"
    }

    invalid_rows = []

    print("Чтение файла data.csv...")
    
    
    with open("data.csv", "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=";")
        
        for i, row in enumerate(reader):
            is_valid = True
            
    
            if None in row or len(row) != len(regex_patterns):
                invalid_rows.append(i)
                continue

            for column, pattern in regex_patterns.items():
                value = row.get(column, "")
                if not value or not re.match(pattern, value):
                    is_valid = False
                    break  
            
            if not is_valid:
                invalid_rows.append(i)
                
    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    
    # Считаем контрольную сумму
    checksum = calculate_checksum(invalid_rows)
    print(f"Контрольная сумма: {checksum}")
    
    # Записываем результат
    serialize_result(variant, checksum)
    print("Результаты записаны в result.json")

if __name__ == "__main__":
    main()