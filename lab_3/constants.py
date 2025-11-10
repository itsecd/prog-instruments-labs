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


VALIDATION_PATTERNS = {
    COLUMN_TELEPHONE: re.compile(r'^\+7[\(\-\s]?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{2}[\-\s]?\d{2}$'),
    COLUMN_HEIGHT: re.compile(r'^[12][,\.]\d{2}$'),
    COLUMN_INN: re.compile(r'^\d{12}$'),
    COLUMN_IDENTIFIER: re.compile(r'^\d{2}[-_\s]?\d{2}[-_\s]?\/?\d{2}$'),
    COLUMN_LATITUDE: re.compile(r'^-?\d{1,2}\.\d+$'),
    COLUMN_BLOOD_TYPE: re.compile(r'^(A|B|AB|O)\s*[+\-−]$'),
    COLUMN_ISSN: re.compile(r'^\d{4}-\d{4}$'),
    COLUMN_UUID: re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    COLUMN_DATE: re.compile(r'^\d{4}[-_\/]\d{2}[-_\/]\d{2}$')
}

COLUMNS_TO_VALIDATE = list(VALIDATION_PATTERNS.keys())
VARIANT = 72
RESULT_FILENAME = 'result.json'