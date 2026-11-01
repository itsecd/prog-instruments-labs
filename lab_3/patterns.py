"""Регулярные выражения для проверки значений столбцов CSV-файла."""

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"[a-z0-9]+(?:[._-][a-z0-9]+)*@[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,}"
    ),
    "height": re.compile(r"(?:0\.[5-9]|1\.\d|2\.[0-4])\d"),
    "snils": re.compile(r"\d{11}"),
    "passport": re.compile(r"\d{2} \d{2} \d{6}"),
    "occupation": re.compile(r"[А-ЯЁа-яёA-Za-z]+(?:[ -][А-ЯЁа-яёA-Za-z]+)*"),
    "longitude": re.compile(r"-?(?:180\.0+|(?:1[0-7]\d|[1-9]?\d)\.\d+)"),
    "hex_color": re.compile(r"#[0-9a-fA-F]{6}"),
    "issn": re.compile(r"\d{4}-\d{3}[\dX]"),
    "locale_code": re.compile(r"[a-z]{2,3}(?:-[a-z]{2,4})?"),
    "time": re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{6}"),
}
