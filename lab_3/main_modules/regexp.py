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
