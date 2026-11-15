# validators.py
import re

VALIDATORS = {
    "email": re.compile(
        r"^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*@"
        r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+$"
    ),
    "telephone": re.compile(r"^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$"),
    "height": re.compile(r"^(?:1\.\d{2}|2\.00)$"),
    "snils": re.compile(r"^[0-9]{11}$"),
    "identifier": re.compile(r"^\d{2}-\d{2}/\d{2}$"),
    "occupation": re.compile(
        r"^[A-Za-zА-Яа-яЁё]+(?:[ -][A-Za-zА-Яа-яЁё]+)*$"
    ),
    "longitude": re.compile(r"-?(?:180(?:\.0+)?|(?:1[0-7][0-9]|"
                            r"[0-9]{1,2})(?:\.[0-9]+)?)"),
    "blood_type": re.compile(r"^(?:A|B|AB|O)[+\u2212]$"),
    "issn": re.compile(r"^\d{4}-\d{4}$"),
    "locale_code": re.compile(r"^[a-z]{2,3}(?:-[a-z]{4})?"
                              r"(?:-(?:[a-z]{2}|[0-9]{3}))?$"),
    "date": re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
}


def find_invalid_fields(row: dict, validators: dict = VALIDATORS) -> list[str]:
    """Возвращает имена полей, значения которых не прошли валидацию.

    :param row: словарь {имя_колонки: значение}
    :param validators: словарь {имя_колонки: скомпилированная регулярка}
    :raises KeyError: если для поля нет валидатора (ошибка конфигурации)
    """
    invalid = []
    for field, value in row.items():
        if field is None:
            invalid.append("<extra>")
            continue
        pattern = validators.get(field)
        if pattern is None:
            raise KeyError(
                f"Нет валидатора для поля {field!r}. "
                f"Ключи: {sorted(validators)}"
            )
        if not pattern.fullmatch((value or "").strip()):
            invalid.append(field)
    return invalid


def validate_row(row: dict, validators: dict = VALIDATORS) -> bool:
    """True, если все поля строки прошли валидацию."""
    return not find_invalid_fields(row, validators)
