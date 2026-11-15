"""Regular expressions for the ten fields in variant 20."""

import re

PATTERNS = {
    # +7-(XXX)-XXX-XX-XX
    "telephone": re.compile(
        r"\+7-\([0-9]{3}\)-[0-9]{3}-[0-9]{2}-[0-9]{2}"
    ),
    # Рост: 1.xx или 2.xx, ровно 2 знака после точки.
    "height": re.compile(r"[12]\.[0-9]{2}"),
    # СНИЛС: 11 цифр.
    "snils": re.compile(r"[0-9]{11}"),
    # identifier: XX-XX/XX.
    "identifier": re.compile(r"[0-9]{2}-[0-9]{2}/[0-9]{2}"),
    # occupation: буквы, пробелы и дефисы (без цифр и подчёркиваний).
    "occupation": re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z -]*"),
    # Долгота: -180..180, дробная часть — до 6 знаков.
    "longitude": re.compile(
        r"-?(?:(?:[0-9]|[1-9][0-9]|1[0-7][0-9])(?:\.[0-9]{1,6})?"
        r"|180(?:\.0{1,6})?)"
    ),
    # Группа крови: A/B/AB/O + '+' или '−' (U+2212) или '-'.
    "blood_type": re.compile(r"(?:A|B|AB|O)[+\u2212-]"),
    # ISSN: XXXX-XXXX.
    "issn": re.compile(r"[0-9]{4}-[0-9]{4}"),
    # locale_code: xx или xx-xx.
    "locale_code": re.compile(r"[a-z]{2}(?:-[a-z]{2})?"),
    # date: YYYY-MM-DD.
    "date": re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}"),
}