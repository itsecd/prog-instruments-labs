import re

TELEPHONE_PATTERN = r"^\+7-\([0-9]{3}\)-[0-9]{3}-[0-9]{2}-[0-9]{2}$"
HTTP_STATUS_PATTERN = r"^(?:1|2|3|4|5)\d{2} [A-Za-z][A-Za-z0-9 \-]*$"
SNILS_PATTERN = r"^\d{11}$"
IDENTIFIER_PATTERN = r"^\d{2}-\d{2}/\d{2}$"
IPV4_PATTERN = r"^(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})$"
LONGITUDE_PATTERN = r"^(-?(?:180(?:\.0{1,6})?|1[0-7]\d(?:\.\d{1,6})?|[0-9]?\d(?:\.\d{1,6})?))$"
BLOOD_TYPE_PATTERN = r"^(A|B|AB|O)[+\u2212]$"
ISBN_PATTERN = r"^(?:\d{1,3}-)?\d{1}-?\d{4,5}-?\d{3,4}-?\d$"
LOCALE_CODE_PATTERN = r"^[a-z]{2}(-[a-z]{2,4})?$"
DATE_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"

DEFAULT_FILE_PATH = "18.csv"
DEFAULT_VARIANT = 18
CSV_DELIMITER = ";"


def get_validation_patterns() -> list[re.Pattern]:
    patterns = [
        TELEPHONE_PATTERN,
        HTTP_STATUS_PATTERN,
        SNILS_PATTERN,
        IDENTIFIER_PATTERN,
        IPV4_PATTERN,
        LONGITUDE_PATTERN,
        BLOOD_TYPE_PATTERN,
        ISBN_PATTERN,
        LOCALE_CODE_PATTERN,
        DATE_PATTERN
    ]
    return [re.compile(pattern) for pattern in patterns]