import csv
import re
import checksum

patterns = {
    "telephone":re.compile(r"^(?:\+7|8)-[(][0-9]{3}[)]-[0-9]{3}-[0-9]{2}-[0-9]{2}$"),
    "height":re.compile(r"^[0-9][.][0-9]{2}$"),
    "snils":re.compile(r"^[0-9]{11}$"),
    "identifier":re.compile(r"^[0-9]{2}-[0-9]{2}[/][0-9]{2}$"),
    "occupation":re.compile(r"^[a-zA-Zа-яА-Я- ]+$"),
    "longitude":re.compile(r"^-?([0-9]|[1-9][0-9]|1[0-7][0-9])[.][0-9]{1,6}$"),
    "blood_type":re.compile(r"^(O|A|B|AB)[+\u2212]$"),
    "issn":re.compile(r"^[0-9]{4}-[0-9]{4}$"),
    "locale_code":re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$"),
    "date":re.compile("^[0-9]{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])$"),
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
        i=0
        for row in reader:
            if not is_valid(row):
                i+=1
                wrong_rows.append(i)
    check= checksum.calculate_checksum(wrong_rows)
    checksum.serialize_result(12,check)

if __name__=='__main__':
    main()