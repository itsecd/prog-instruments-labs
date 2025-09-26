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
