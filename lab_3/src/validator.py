import re
import ipaddress
import pandas as pd
from datetime import datetime

from validator_settings import VALIDATION_RULES


def validate_regex(value, rule):
    return bool(re.fullmatch(rule["pattern"], str(value)))


def validate_ip(value, rule):
    try:
        ipaddress.IPv4Address(value)
        return True
    except Exception:
        return False


def validate_range(value, rule):
    try:
        val = float(value)
        return rule["min"] <= val <= rule["max"]
    except ValueError:
        return False


def validate_enum(value, rule):
    return str(value).strip() in rule["values"]


def validate_date(value, rule):
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return rule["min_year"] <= dt.year <= rule["max_year"]
    except ValueError:
        return False


VALIDATORS = {
    "regex": validate_regex,
    "ip_address": validate_ip,
    "range": validate_range,
    "enum": validate_enum,
    "date_logic": validate_date,
}


def validate_field(field_name, value):
    rule = VALIDATION_RULES[field_name]
    validator = VALIDATORS[rule["type"]]
    return validator(value, rule)


def validate_row(row):
    return all(validate_field(col, row[col]) for col in VALIDATION_RULES.keys())
