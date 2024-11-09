import re
import csv

from checksum import calculate_checksum, serialize_result

PATTERNS = {
    "email": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "http_status_message":"^\d{3} [A-Za-z\s]+$",
    "inn": "^\d{12}$",
    "passport": "^\d{2} \d{2} \d{6}$",
    "ip_v4":"^(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$",
    "latitude": "^(-?[1-8]?\d(?:\.\d{1,})?|90(?:\.0{1,})?)$",
    "hex_color": "^#[0-9a-fA-F]{6}$",
    "isbn": "\\d+-\\d+-\\d+-\\d+(:?-\\d+)?$",
    "uuid": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$",
    "time": "^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.(\d{1,6})$"}

def check_data(row: list) -> bool:
    '''Проверяет строку таблицы на валидность
    Args:
        row (list): Строка таблицы для проверки
    Returns:
        bool: Результат прохождения/непрохождения проверки
    '''
    for key, value in zip(PATTERNS.keys(), row):
        if not re.match(PATTERNS[key], value):
            return False
    return True


def find_invalid_data(data: list) -> None:
    '''Поиск индексов невалидных данных
    Args:
        data (list): Массив строк с данными

    Returns: None
    '''
    list_index = []
    index = 0
    for elem in data:
        if not check_data(elem):
            list_index.append(index)
        index += 1
    serialize_result(21, calculate_checksum(list_index))


def read_csv(file_name: str) -> list:
    '''Десериализация данных из csv файла'''
    list_data = []
    with open(file_name, "r", newline="", encoding="utf-16") as file:
        reader = csv.reader(file, delimiter=";")
        for elem in reader:
            list_data.append(elem)
    list_data.pop(0)
    return list_data


if __name__ == "__main__":
    find_invalid_data(read_csv("21.csv"))