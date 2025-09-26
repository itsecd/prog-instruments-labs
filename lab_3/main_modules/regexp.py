from pandas import DataFrame
import re


def validata_phone_number(phone_number: str) -> bool:
    """
    Функция проверки номера телефона
    :param phone_number: номер телефона
    :return: Подходит ли номер под параметры True/False
    """
    pattern = r'^\+7-\((\d{3})\)-(\d{3})-(\d{2})-(\d{2})$'
    if re.fullmatch(pattern, phone_number):
        return True
    return False


def validate_http_status(http_status: str) -> bool:
    pattern = r'^(100|1[0-9][0-9]|2[0-9][0-9]|3[0-9][0-9]|4[0-9][0-9]|5[0-9][0-9])\s+[A-Za-z].+'
    if re.fullmatch(pattern, http_status):
        return True
    return False


def validate_inn(inn: str) -> bool:
    pattern = r'^\d{10}$|^\d{12}$'
    if re.fullmatch(pattern, inn):
        return True
    return False


def validate_identifier(identifier: str) -> bool:
    pattern = r'^\d{2}-\d{2}/\d{2}$'
    if re.fullmatch(pattern, identifier):
        return True
    return False


def validate_ipv4(ipv4: str) -> bool:
    pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    if re.fullmatch(pattern, ipv4):
        return True
    return False


def validate_latitude(latitude: str) -> bool:
    pattern = r'^[-+]?([0-8]?[0-9]|90)(\.\d+)?$'
    if re.fullmatch(pattern, latitude):
        return True
    return False


def validate_blood_type(blood_type: str) -> bool:
    pattern = r'^(A|B|AB|O)[+−]\s*$'
    if re.fullmatch(pattern, blood_type):
        return True
    return False


def validate_isbn(isbn: str) -> bool:
    pattern = r'^((\d{1})-(\d{5})-(\d{3})-(\d{1}))$|^(\d{3})-(\d{1})-(\d{5})-(\d{3})-(\d{1})$'
    if re.fullmatch(pattern, isbn):
        return True
    return False


def validate_uuid(uuid: str) -> bool:
    pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    if re.fullmatch(pattern, uuid):
        return True
    return False


def validate_date(date: str) -> bool:
    pattern = r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'
    if re.fullmatch(pattern, date):
        return True
    return False


def validate_by_pattern(data: str, pattern: str) -> bool:
    sub_pattern = rf'{pattern}'
    if re.fullmatch(sub_pattern, data):
        return True
    return False


def get_rows_with_mistakes(data_frame: DataFrame, patterns) -> list:
    rows_with_mistakes = []
    patter_index = 0
    for column_name in data_frame:
        for row_index, value in data_frame[column_name].items():
            if not validate_by_pattern(value, patterns[patter_index]):
                if row_index not in rows_with_mistakes:
                    rows_with_mistakes.append(row_index)
        patter_index += 1
    return rows_with_mistakes
