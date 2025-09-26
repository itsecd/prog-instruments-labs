from main_modules.checksum import serialize_result
from main_modules.file_work import read_json, read_csv
from main_modules.regexp import (validata_phone_number, validate_http_status, validate_inn, validate_identifier,
                                 validate_ipv4, validate_latitude, validate_blood_type, validate_isbn, validate_uuid,
                                 validate_date, validate_by_pattern, get_rows_with_mistakes)

if __name__ == "__main__":
    settings = read_json("settings.json")
    patterns = read_json(settings["path_to_patterns"])
    keys = list(patterns.keys())
    data = read_csv(settings["path_to_data"])
    rows_with_mistakes = get_rows_with_mistakes(data, keys)
    print(rows_with_mistakes)
    print(len(rows_with_mistakes))



