VARIANT = 18
REGULAR_DICT = {
    "telephone": r"^\+7\-\(\d{3}\)\-\d{3}\-\d{2}\-\d{2}$",
    "http_status_message": r"^\d{3} ",
    "snils": r"^\d{11}$",
    "identifier": r"^\d{2}\-\d{2}\/\d{2}$",
    "ip_v4": r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$",
    "longitude": r"^\-?(180\.0{1,6}|(1[0-7]\d|[1-9]?\d)\.\d{1,6})$",
    "blood_type": r"^([ABO]|AB)[\u2212+]$",
    "isbn": r"(\d\d\d\-)?\d\-\d{5}\-\d{3}\-\d",
    "locale_code": r"^[a-z]{2,3}(-[a-z]{2})?$",
    "date": r"^\d{4}\-(0\d|1[0-2])\-([0-2]\d|3[01])$"
}
