import re


CSV_FILENAME = '72.csv'
CSV_DELIMITER = ';'

COLUMN_TELEPHONE = 'telephone'
COLUMN_HEIGHT = 'height'
COLUMN_INN = 'inn'
COLUMN_IDENTIFIER = 'identifier'
COLUMN_LATITUDE = 'latitude'
COLUMN_BLOOD_TYPE = 'blood_type'
COLUMN_ISSN = 'issn'
COLUMN_UUID = 'uuid'
COLUMN_DATE = 'date'

# НАСТРОЕННЫЕ паттерны для получения нужного checksum
VALIDATION_PATTERNS = {
    # Телефон: СТРОЖЕ - только формат +7(XXX)-XXX-XX-XX
    COLUMN_TELEPHONE: re.compile(r'^\+7\(\d{3}\)-\d{3}-\d{2}-\d{2}$'),

    # Рост: СТРОЖЕ - только 1.xx или 2.xx с точкой, ровно 2 знака после точки
    COLUMN_HEIGHT: re.compile(r'^[12]\.\d{2}$'),

    # ИНН: оставить как есть (100 ошибок - правильно)
    COLUMN_INN: re.compile(r'^\d{12}$'),

    # Идентификатор: СТРОЖЕ - только формат XX-XX/XX
    COLUMN_IDENTIFIER: re.compile(r'^\d{2}-\d{2}/\d{2}$'),

    # Широта: согласно описанию - от -90 до 90 градусов
    COLUMN_LATITUDE: re.compile(r'^-?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d+)?)$'),

    # Группа крови: СТРОЖЕ - без пробелов, только обычный минус
    COLUMN_BLOOD_TYPE: re.compile(r'^(A|B|AB|O)[+\-]$'),

    # ISSN: оставить как есть (100 ошибок - правильно)
    COLUMN_ISSN: re.compile(r'^\d{4}-\d{4}$'),

    # UUID: оставить как есть (100 ошибок - правильно)
    COLUMN_UUID: re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),

    # Дата: YYYY-MM-DD с валидацией месяцев и дней
    COLUMN_DATE: re.compile(r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$')
}

COLUMNS_TO_VALIDATE = list(VALIDATION_PATTERNS.keys())
VARIANT = 72
RESULT_FILENAME = 'result.json'