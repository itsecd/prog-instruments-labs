import csv
import re
import pandas as pd


from checksum import calculate_checksum


regular = {
    "telephone": "^\\+7-\\(9\\d{2}\\)-\\d{3}-\\d{2}-\\d{2}$",  # telephone
    "height": "^[1-2]\\.\\d{2}$",  # height
    "snils": "^\\d{11}$",  # snils
    "identifier": "^\\d{2}-\\d{2}/\\d{2}$",  # identifier
    "occupation": "^[а-яА-ЯёЁa-zA-Z\\s-]+$",  # occupation
    "longitude": "^-?(180(\\.0+)?|1[0-7]\\d(\\.\\d+)?|[1-9]?\\d(\\.\\d+)?)$",  # longitude
    "blood_type": "^(?:AB|A|B|O)[+\\u2212]$",  # blood type
    "issn": "^\\d{4}-\\d{4}$",  # issn
    "locale_code": "^[a-zA-Z]+(-[a-zA-Z]+)*$",  # local code
    "date": "^\\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|1\\d|2[0-9]|3[0-1])$"  # date
}


def open_csv(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, encoding="utf-16", sep=";")
        return data
    except Exception as e:
        print(f"Ошибка при открытия файла {path}: {e}")
        raise


def validate_row(row):
    for name_column, value in row.items():
        if name_column in regular:
            pattern = regular[name_column]
            if not re.match(pattern, str(value)):
                return False
    return True


def process_csv(file_path):
    data = open_csv(file_path)
    invalid_row_numbers = []
    for i, row in data.iterrows():
        if not validate_row(row):
            invalid_row_numbers.append(i)
    return invalid_row_numbers


def main():
    csv_file_path = r'E:\TMP\TMP\lab_3\4.csv'
    invalid_rows = process_csv(csv_file_path)
    invalid_rows.sort()
    checksum = calculate_checksum(invalid_rows)
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()