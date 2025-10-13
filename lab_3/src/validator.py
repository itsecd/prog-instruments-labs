import re
import ipaddress
import pandas as pd
from datetime import datetime
from typing import Any

from validator_settings import VALIDATION_RULES


def validate_regex(value: Any, rule: dict) -> bool:
    """
    Проверяет значение с помощью регулярного выражения.

    :param value: Значение для проверки
    :param rule: Словарь с ключом "pattern" для регулярного выражения
    :return: True, если значение соответствует шаблону, иначе False
    """
    return bool(re.fullmatch(rule["pattern"], str(value)))


def validate_ip(value: Any) -> bool:
    """
    Проверяет, является ли значение корректным IPv4 адресом.
    :param value: Значение для проверки
    :return: True, если значение корректный IPv4, иначе False
    """
    try:
        ipaddress.IPv4Address(value)
        return True
    except Exception:
        return False


def validate_range(value: Any, rule: dict) -> bool:
    """
    Проверяет, находится ли числовое значение в указанном диапазоне.

    :param value: Значение для проверки
    :param rule: Словарь с ключами "min" и "max" для диапазона
    :return: True, если значение в диапазоне, иначе False
    :param value:
    :param rule:
    :return:
    """
    try:
        val = float(value)
        return rule["min"] <= val <= rule["max"]
    except ValueError:
        return False


def validate_enum(value: Any, rule: dict) -> bool:
    """
    Проверяет, входит ли значение в набор допустимых значений.

    :param value: Значение для проверки
    :param rule: Список допустимых значений
    :return: True, если значение допустимо, иначе False
    """
    return str(value).strip() in rule["values"]


def validate_date(value: Any, rule: dict) -> bool:
    """
    Проверяет корректность даты и диапазон года.
    :param value: Cтрока с датой в формате YYYY-MM-DD
    :param rule: Cловарь с ключами "min_year" и "max_year"
    :return: True, если дата корректна и год в диапазоне, иначе False
    """
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return rule["min_year"] <= dt.year <= rule["max_year"]
    except ValueError:
        return False


VALIDATORS: dict[str, Any] = {
    "regex": validate_regex,
    "ip_address": validate_ip,
    "range": validate_range,
    "enum": validate_enum,
    "date_logic": validate_date,
}


def validate_field(field_name: str, value: Any) -> bool:
    """
    Универсальная проверка одного поля по правилу из VALIDATION_RULES.
    :param field_name: Имя поля из CSV
    :param value: Значение поля для проверки
    :return: True, если значение корректно, иначе False
    """
    rule = VALIDATION_RULES[field_name]
    validator = VALIDATORS[rule["type"]]
    return validator(value, rule)


def validate_row(row: pd.Series) -> bool:
    """
    Проверяет корректность всех полей строки DataFrame.
    :param row: Cтрока DataFrame
    :return: True, если все поля корректны, иначе False
    """
    return all(validate_field(col, row[col]) for col in VALIDATION_RULES.keys())
