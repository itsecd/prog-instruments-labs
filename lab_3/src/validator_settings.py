VALIDATION_RULES = {
    "telephone": {
        "type": 'regex',
        "pattern":  r'^\+7-\(\d{3}\)-\d{3}-\d{2}-\d{2}$',
        "error": 'Номер телефона должен иметь такой формат: +7-(XXX)-XXX-XX-XX'
    },

    "http_status_message": {
        "type": 'regex',
        "pattern": r'^\d{3} [A-za-z]+$',
        "error": 'HTTP статус должен иметь такой формат: "Код Описание"'
    },

    "inn": {
        "type": 'regex',
        "pattern": r'^\d{12}$',
        "error": "ИНН должен содержать 12 цифр, которые не разделены никакими символами"
    },

    "identifier": {
        
    }
}