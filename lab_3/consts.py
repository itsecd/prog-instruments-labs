EMAIL_PATTERN = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
HTTP_STATUS_PATTERN = r"^(?:1|2|3|4|5)\d{2} [A-Za-z][A-Za-z0-9 \-]*$"
INN_PATTERN = r"^(?:\d{10}|\d{12})$"
PASSPORT_PATTERN = r"^\d{2} \d{2} \d{6}$"
IPV4_PATTERN = (
    r"^(?:"
    r"(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d{1,2})$"
)
LATITUDE_PATTERN = r"^[+-]?(?:90(?:\.0+)?|(?:[0-8]?\d(?:\.\d+)?))$"
HEX_COLOR_PATTERN = r"^#[0-9A-Fa-f]{6}$"
ISBN_PATTERN = r"^(?=(?:.*\d){13}$)[\d-]+$"
UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
TIME_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?$"


DEFAULT_FILE_PATH = "13.csv"
DEFAULT_VARIANT = 13
FILE_ENCODING = "windows-1251"