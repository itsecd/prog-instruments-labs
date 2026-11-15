# validators.py
import re

VALIDATORS = {
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "telephone": re.compile(r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$"),
    "height": re.compile(r"^(?:[01]\.\d{2}|2\.[0-6]\d|2\.7[0-5])$"),
    "snils": re.compile(r"^[0-9]{11}$"),
    "identifier": re.compile(r"^\d{2}\-\d{2}\/\d{2}$"),
    "occupation": re.compile(r"^[а-яА-ЯёЁa-zA-Z -]+$"),
    "longitude": re.compile(r"^-?(?:\d\.\d{0,}|\d{2}\.\d{0,}|"
                            r"1[0-7]\d\.\d{0,}|180)$"),
    "blood_type": re.compile(r"^(?:O|A|B|AB)[+\u2212]$"),
    "issn": re.compile(r"^\d{4}-\d{4}$"),
    "locale_code": re.compile(r"^[a-z]{2,3}(?:-[a-z]{4})?"
                              r"(?:-(?:[a-z]{2}|[0-9]{3}))?$"),
    "date": re.compile(
        r"[0-9]{4}-(?:"
        r"(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])"   # 31-дневные
        r"|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)"          # 30-дневные
        r"|02-(?:0[1-9]|1[0-9]|2[0-8])"                     # февраль 1–28
        r"|02-29"                                            # 29 февраля
        r")")
}


def validate_row(row: dict, validators: dict = VALIDATORS) -> list[str]:
    invalid = []
    for field, value in row.items():
        if field is None:   # DictReader кладёт None для «лишних» колонок
            invalid.append("<extra>")
            continue
        pattern = validators.get(field)
        if pattern is None:
            raise KeyError(f"Нет валидатора для поля {field!r}. "
                           f"Ключи: {sorted(validators)}")
        if not pattern.fullmatch((value or "").strip()):
            invalid.append(field)
    return invalid
