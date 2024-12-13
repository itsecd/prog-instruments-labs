PATH_TO_CSV = "lab_3/table.csv"
VARIANT = 48

REGEX_PATTERNS = {
    "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
    "http_status_message": r"^\d{3} ",
    "snils": r"^\d{11}$",
    "passport": r"^\d{2} \d{2} \d{6}$",
    "ip_v4": r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$",
    "longitude": r"^\-?(180\.0{1,6}|(1[0-7]\d|[1-9]?\d)\.\d{1,6})$", 
    "hex_color": r"^#[a-fA-F0-9]{6}$",
    "isbn": r"(\d\d\d\-)?\d\-\d{5}\-\d{3}\-\d",
    "locale_code": r"^[a-z]{2,3}(-[a-z]{2})?$",
    "time": r"^\d{2}:\d{2}:\d{2}\.\d+$"
}