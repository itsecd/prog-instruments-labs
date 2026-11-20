import re

REGEXES = {
  "email": re.compile(r"^[A-Za-z0-9.]+@[A-Za-z0-9.]+\.[A-Za-z]+$"),
  "height": re.compile(r"^[12]\.\d\d$"),
  "snils": re.compile(r"^\d{11}$"),
  "passport": re.compile(r"^\d\d\s\d\d\s\d{6}$"),
  "occupation": re.compile(r"^[A-Za-zА-Яа-яЁё\s-]+$"),
  "longitude": re.compile(r"^-?(?:1[0-7][0-9]|180|[0-9]{1,2})(?:\.\d+)?$"),
  "hex_color": re.compile(r"^#[A-Fa-f0-9]{6}$"),
  "issn": re.compile(r"^[0-9]{4}-[0-9]{4}$"),
  "locale_code": re.compile(r"^[a-z]{2,3}(-[a-z]{2})?$"),
  "time": re.compile(r"^(?:[0-1][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{6}$")
}