import re
CSV_FILENAME = '69.csv'
CSV_DELIMITER = ';'
COLUMN_EMAIL = 'email'
COLUMN_HTTP_STATUS = 'http_status_message'
COLUMN_INN = 'inn'
COLUMN_PASSPORT = 'passport'
COLUMN_IP_V4 = 'ip_v4'
COLUMN_LATITUDE = 'latitude'
COLUMN_HEX_COLOR = 'hex_color'
COLUMN_ISBN = 'isbn'
COLUMN_UUID = 'uuid'
COLUMN_TIME = 'time'
VALIDATION_PATTERNS = {
    COLUMN_EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    COLUMN_HTTP_STATUS: re.compile(r'^\d{3} [A-Z][A-Za-z ]+$'),
    COLUMN_IP_V4: re.compile(r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
    COLUMN_HEX_COLOR: re.compile(r'^#[0-9A-Fa-f]{6}$'),
    COLUMN_INN: re.compile(r'^\d{12}$'),
    COLUMN_PASSPORT: re.compile(r'^\d{2} \d{2} \d{6}$'),
    COLUMN_UUID: re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    COLUMN_LATITUDE: re.compile(r'^(-?(?:90(?:\.0{1,6})?|[1-8]?\d(?:\.\d{1,6})?))$'),
    COLUMN_ISBN: re.compile(r'^(\d{3}-)?\d-\d{5}-\d{3}-\d$'),
    COLUMN_TIME: re.compile(r'^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.(\d{1,6})$')
}
COLUMNS_TO_VALIDATE = list(VALIDATION_PATTERNS.keys())
VARIANT = 69
RESULT_FILENAME = 'result.json'