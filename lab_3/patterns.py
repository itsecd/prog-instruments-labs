import re

OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"

PATTERNS = {
    "email": re.compile(
        r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*@(?:[A-Za-z0-9]+\.)+[A-Za-z]{2,}"
    ),
    "http_status_message": re.compile(r"[1-5][0-9]{2} [A-Za-z]+(?:[ '-][A-Za-z]+)*"),
    "snils": re.compile(r"[0-9]{11}"),
    "passport": re.compile(r"[0-9]{2} [0-9]{2} [0-9]{6}"),
    "ip_v4": re.compile(rf"{OCTET}(?:\.{OCTET}){{3}}"),
    "longitude": re.compile(
        r"-?(?:(?:1[0-7][0-9]|[1-9]?[0-9])(?:\.[0-9]+)?|180(?:\.0+)?)"
    ),
    "hex_color": re.compile(r"#[0-9A-Fa-f]{6}"),
    "isbn": re.compile(r"(?:[0-9]{3}-)?[0-9]-[0-9]{5}-[0-9]{3}-[0-9X]"),
    "locale_code": re.compile(r"[a-z]{2}(?:-[a-z]{2})?"),
    "time": re.compile(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{6}"),
}
