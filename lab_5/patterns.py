"""Regular expressions for the ten fields in variant 26."""

import re

# Each IPv4 octet is an integer from 0 to 255 without leading zeros.
IPV4_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"

# Patterns describe the synthetic data format specified by the assignment.
PATTERNS = {
    "telephone": re.compile(r"\+7-\([0-9]{3}\)-[0-9]{3}-[0-9]{2}-[0-9]{2}"),
    "http_status_message": re.compile(r"[1-5][0-9]{2} [A-Za-z]+(?:[ -][A-Za-z]+)*"),
    "snils": re.compile(r"[0-9]{11}"),
    "identifier": re.compile(r"[0-9]{2}-[0-9]{2}/[0-9]{2}"),
    "ip_v4": re.compile(rf"(?:{IPV4_OCTET}\.){{3}}{IPV4_OCTET}"),
    "longitude": re.compile(
        r"-?(?:(?:[0-9]|[1-9][0-9]|1[0-7][0-9])(?:\.[0-9]+)?"
        r"|180(?:\.0+)?)"
    ),
    "blood_type": re.compile(r"(?:A|B|AB|O)[+\u2212]"),
    "isbn": re.compile(
        r"(?:[0-9]-[0-9]{5}-[0-9]{3}-[0-9X]"
        r"|[0-9]{3}-[0-9]-[0-9]{5}-[0-9]{3}-[0-9])"
    ),
    "locale_code": re.compile(r"[a-z]{2}(?:-[a-z]{2})?"),
    "date": re.compile(r"\d{4}-\d{2}-\d{2}"),
}
