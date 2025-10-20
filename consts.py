# Regex patterns for data validation
NAME_PATTERN = r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$"
SURNAME_PATTERN = r"^[A-Z][a-z]+$"
POSTAL_CODE_PATTERN = r"^\d{2}-\d{3}$"
EMAIL_PATTERN = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
PHONE_PATTERN = r"^\+?\d{1,3}[- ]?\(?\d{2,3}\)?[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}$"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
PASSPORT_PATTERN = r"^[A-Z]{2}\d{6}$"
DECIMAL_PATTERN = r"^\d+\.\d{2}$"
ID_PATTERN = r"^[A-Za-z0-9]{8,}$"
URL_PATTERN = r"^(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

# File and application settings
DEFAULT_FILE_PATH = "13.csv"
DEFAULT_VARIANT = 13
FILE_ENCODING = "windows-1251"