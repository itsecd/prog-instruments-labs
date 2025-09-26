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
