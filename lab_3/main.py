import csv
import re
import json
from checksum import calculate_checksum

# --- Константы ---
CSV_FILE_PATH = "10.csv"
VARIANT_NUMBER = 10

# --- Словарь с регулярными выражениями для каждого поля ---
REGEX_PATTERNS = {
    "telephone": r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$",
    "http_status_message": r"^\d{3}\s[A-Z][a-zA-Z\s]*$",
    "snils": r"^\d{11}$",
    "identifier": r"^\d{2}-\d{2}\/\d{2}$",
    "ip_v4": r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$",
    "longitude": r"^-?(\d{1,2}|1[0-7]\d|180)(\.\d+)?$",
    "blood_type": r"^(A|B|AB|O)[\+\-\u2212]$",
    "isbn": r"^(\d+[-]){3,4}[\dX]$",
    "locale_code": r"^[a-z]{2,3}(-[a-z]{2})?$",
    "date": r"^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
}

# Тут будет основная логика