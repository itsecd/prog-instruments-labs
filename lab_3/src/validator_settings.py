VALIDATION_RULES = {
    "telephone": {
        "type": 'regex',
        "pattern":  r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$'
    },

    "http_status_message": {
        "type": 'regex',
        "pattern": r'^\d{3} [A-Za-z ]+$'
    },

    "inn": {
        "type": 'regex',
        "pattern": r'^\d{12}$'
    },

    "identifier": {
        "type": 'regex',
        "pattern": r'^\d{2}-\d{2}/\d{2}$',
    },

    "ip_v4": {
        "type": 'ip_address'
    },

    "latitude": {
        "type": 'range',
        "min": -90,
        "max": 90
    },

    "blood_type": {
        "type": "enum",
        "values": [
            "A+", "A−",
            "B+", "B−",
            "AB+", "AB−",
            "O+", "O−"
        ]
    },

    "isbn": {
        "type": 'regex',
        "pattern": r'^(?:\d{1,5}-)?\d-\d{5}-\d{3}-\d$|^\d{3}-\d-\d{5}-\d{3}-\d$'
    },

    "uuid": {
        "type": 'regex',
        'pattern': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    },

    "date": {
        "type": 'date_logic',
        "min_year": 1900,
        "max_year": 2025
    }
}