import csv
import re
import checksum

patterns = {
    "telephone": re.compile(r"^\+7-[(][0-9]{3}[)]-[0-9]{3}-[0-9]{2}-[0-9]{2}$"),
    "height": re.compile(r"^[12][.][0-9]{2}$"),
    "snils": re.compile(r"^[0-9]{11}$"),
    "identifier": re.compile(r"^[0-9]{2}-[0-9]{2}[/][0-9]{2}$"),
    "occupation": re.compile(r"^[a-zA-Zа-яА-ЯёЁ-]+(?: [a-zA-Zа-яА-ЯёЁ-]+)*$"),
    "longitude": re.compile(r"^-?([0-9]|[1-9][0-9]|1[0-7][0-9])[.][0-9]{1,6}$"),
    "blood_type": re.compile(r"^(O|A|B|AB)[+\u2212]$"),
    "issn": re.compile(r"^[0-9]{4}-[0-9]{4}$"),
    "locale_code": re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$"),
    "date": re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"),
}

def is_valid(dicti):
    for i in dicti.keys():
        if not patterns[i].match(dicti[i]):
            return False
    return True

def main():
    wrong_rows = []
    with open('12.csv',encoding='utf-16') as file:
        reader = csv.DictReader(file, delimiter=';', quotechar='"')
        for i, row in enumerate(reader):
            if not is_valid(row):
                wrong_rows.append(i)
    check= checksum.calculate_checksum(wrong_rows)
    checksum.serialize_result(12,check)

if __name__=='__main__':
    main()
