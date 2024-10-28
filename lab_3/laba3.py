import csv
import re

from checksum import serialize_result, calculate_checksum


PATTERNS = {
    "email": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
    "height": "^[1-2]\\.\\d{2}$",
    "snils": "^\\d{11}$",
    "passport": "^\\d{2}\\s\\d{2}\\s\\d{6}$",
    "occupation": "^[a-zA-Zа-яА-ЯёЁ\\s-]+$",
    "longitude": "^-?((180(\\.0+)?|1[0-7]?\\d(\\.\\d+)?|\\d{1,2}(\\.\\d+)?)|0(\\.\\d+)?)$",
    "hex_color": "^#([a-f0-9]{6}|[a-f0-9]{3})$",
    "issn": "^\\d{4}-\\d{4}$",
    "locale_code": "^([a-z]{2,3})(?:-([a-zA-Z]{2,4}))?(?:-([a-zA-Z0-9-]+))?$",
    "time": "^([0-1][0-9]|2[0-3])\\:[0-5][\\d]\\:[0-5][\\d]\\.[\\d]{6}"
}


def check_data(row: list) -> bool:
    '''Сheck string for data validity'''
    for key, value in zip(PATTERNS.keys(), row):
        if not re.match(PATTERNS[key], value):
            return False
    return True


import csv
import re

from checksum import serialize_result, calculate_checksum

PATTERNS = {
    "email": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
    "height": "^[1-2]\\.\\d{2}$",
    "snils": "^\\d{11}$",
    "passport": "^\\d{2}\\s\\d{2}\\s\\d{6}$",
    "occupation": "^[a-zA-Zа-яА-ЯёЁ\\s-]+$",
    "longitude": "^-?((180(\\.0+)?|1[0-7]?\\d(\\.\\d+)?|\\d{1,2}(\\.\\d+)?)|0(\\.\\d+)?)$",
    "hex_color": "^#([a-f0-9]{6}|[a-f0-9]{3})$",
    "issn": "^\\d{4}-\\d{4}$",
    "locale_code": "^([a-z]{2,3})(?:-([a-zA-Z]{2,4}))?(?:-([a-zA-Z0-9-]+))?$",
    "time": "^([0-1][0-9]|2[0-3])\\:[0-5][\\d]\\:[0-5][\\d]\\.[\\d]{6}"
}


def check_data(row: list) -> bool:
    '''Проверяет строку на корректность данных по заданным паттернам.

    Аргументы:
    row -- список значений для проверки на соответствие паттернам

    Возвращает:
    bool -- True, если все значения соответствуют паттернам, иначе False
    '''
    for key, value in zip(PATTERNS.keys(), row):
        if not re.match(PATTERNS[key], value):
            return False
    return True


def read_csv(filename: str) -> list:
    '''Считывает данные из CSV файла и возвращает их в виде списка.

        Аргументы:
        filename -- название файла, который необходимо прочитать

        Возвращает:
        list -- список строк, считанных из файла, без заголовка
        '''
    data = []
    with open(filename, "r", newline="", encoding="utf-16") as file:
        reader = csv.reader(file, delimiter=";")
        for el in reader:
            data.append(el)
        data.pop(0)
        return data

def find_incorrect_data(data: list) -> list:
    '''Находит индексы некорректных данных в списке.

        Аргументы:
        data -- список данных, в котором необходимо проверить корректность

        Возвращает:
        list -- список индексов элементов, которые не прошли проверку
        '''
    indexes = []
    index = 0
    for el in data:
        if not check_data(el):
            indexes.append(index)
        index += 1
    return indexes


if __name__ == "__main__":
    filename = "21.csv"
    variant = 21
    data = read_csv(filename)
    indexes = find_incorrect_data(data)
    serialize_result(variant, calculate_checksum(indexes))