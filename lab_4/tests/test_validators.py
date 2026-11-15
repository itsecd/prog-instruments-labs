"""Тесты валидаторов из validators.py."""
import pytest

from src.validators import validate_row


# Базовый словарь со всеми валидными значениями.
# Каждый тест может переопределить одно или несколько полей.
VALID_ROW = {
    "email": "operators.1947@protonmail.com",
    "telephone": "+7-(969)-765-17-05",
    "height": "1.76",
    "snils": "90534478510",
    "identifier": "62-71/26",
    "occupation": "Web-программист",
    "longitude": "-63.65076",
    "blood_type": "AB+",
    "issn": "1931-0891",
    "locale_code": "es-uy",
    "date": "2000-02-14",
}


@pytest.mark.parametrize("field, bad_value", [
    # email
    ("email", "not-an-email"),
    ("email", "a@b..com"),              # двойная точка в домене
    ("email", "a@b"),                   # нет домена верхнего уровня
    # telephone
    ("telephone", "8-(999)-339-96-93"),  # 8 вместо +7
    ("telephone", "+7-935-706-67-89"),  # нет скобок
    ("telephone", "+7-(999)-339-96"),   # не хватает цифр
    # height
    ("height", "3.00"),                 # > 2.75
    ("height", "11.86"),                # не человеческий рост
    ("height", "1,82"),                 # запятая
    ("height", "1.8"),                  # одна цифра после точки
    # snils
    ("snils", "637989665903"),          # 12 цифр
    ("snils", "9053447851"),            # 10 цифр
    # identifier
    ("identifier", "14 62/44"),         # пробел
    ("identifier", "90-8673"),          # нет /
    ("identifier", "04_27/90"),         # подчёркивание
    # occupation
    ("occupation", "08Геолог70"),       # цифры
    ("occupation", "_Реабилитолог_"),   # подчёркивания
    ("occupation", "Инженер!"),         # спецсимвол
    # longitude
    ("longitude", "50,616351"),         # запятая
    ("longitude", "180.5"),             # > 180
    ("longitude", "-180.1"),            # < -180
    ("longitude", "12."),               # точка без дробной части
    # blood_type
    ("blood_type", "A -"),              # пробел вместо +/−
    ("blood_type", "AB0"),              # 0 вместо +/−
    ("blood_type", "Z−"),               # Z не группа ABO
    # issn
    ("issn", "19247875"),               # нет дефиса
    ("issn", "2357_0557"),              # подчёркивание
    # locale_code
    ("locale_code", "5_mk_0"),          # цифры и подчёркивания
    ("locale_code", "ar-eg_0"),         # подчёркивание
    # date
    ("date", "32019-10-24"),            # лишняя цифра
    ("date", "2009/10/11"),             # слэши
    ("date", "2018_09_22"),             # подчёркивания
])
def test_validate_row_rejects_bad_value(field, bad_value):
    """Одно битое поле делает всю строку невалидной."""
    row = {**VALID_ROW, field: bad_value}
    assert validate_row(
        row) is False, f"должно быть невалидно: {field}={bad_value!r}"


@pytest.mark.parametrize("field, good_value", [
    # граничные валидные значения
    ("longitude", "180"),
    ("longitude", "-180"),
    ("longitude", "180.0"),
    ("longitude", "0"),
    ("longitude", "-0.625823"),
    ("date", "2024-02-29"),             # високосный
    ("date", "2000-01-01"),            # X в контрольной цифре
    ("blood_type", "O+"),
    ("blood_type", "O\u2212"),          # минус U+2212
    ("locale_code", "en"),
    ("email", "relate1878@sub.domain.ru"),
])
def test_validate_row_accepts_good_value(field, good_value):
    """Граничные валидные значения проходят."""
    row = {**VALID_ROW, field: good_value}
    assert validate_row(
        row) is True, f"должно быть валидно: {field}={good_value!r}"


def test_validate_row_all_valid():
    """Строка со всеми валидными полями — валидна."""
    assert validate_row(VALID_ROW) is True


def test_validate_row_unknown_field_raises():
    """Неизвестное поле → KeyError (ошибка конфигурации, а не данных)."""
    row = {**VALID_ROW, "unknown_field": "value"}
    with pytest.raises(KeyError):
        validate_row(row)


def test_validate_row_extra_column_returns_false():
    """DictReader кладёт None-ключ для «лишних» колонок — строка невалидна."""
    row = {**VALID_ROW, None: ["extra1", "extra2"]}
    assert validate_row(row) is False
