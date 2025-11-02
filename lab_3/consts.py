import re

EMAIL_PATTERN = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
HTTP_STATUS_PATTERN = r"^(?:1|2|3|4|5)\d{2} [A-Za-z][A-Za-z0-9 \-]*$"
INN_PATTERN = r"^\d{12}$"
PASSPORT_PATTERN = r"^\d{2} \d{2} \d{6}$"
IPV4_PATTERN = (
    r"^(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})$"
)
LATITUDE_PATTERN = r"^(-?(?:90(?:\.0{1,6})?|[1-8]?\d(?:\.\d{1,6})?))$"
HEX_COLOR_PATTERN = r"^#[0-9A-Fa-f]{6}$"
ISBN_PATTERN = r"^(\d{3}-)?\d-\d{5}-\d{3}-\d$"
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
TIME_PATTERN = r"^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.(\d{1,6})$"

DEFAULT_FILE_PATH = "13.csv"
DEFAULT_VARIANT = 13
FILE_ENCODING = "utf-16"
CSV_DELIMITER = ";"

def get_validation_patterns() -> list[re.Pattern]:
    """Returns a list of compiled templates for verification."""
    patterns = [
        EMAIL_PATTERN,
        HTTP_STATUS_PATTERN,
        INN_PATTERN,
        PASSPORT_PATTERN,
        IPV4_PATTERN,
        LATITUDE_PATTERN,
        HEX_COLOR_PATTERN,
        ISBN_PATTERN,
        UUID_PATTERN,
        TIME_PATTERN
    ]
    return [re.compile(pattern) for pattern in patterns]
