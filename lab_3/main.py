from main_modules.checksum import serialize_result
from main_modules.file_work import read_json, read_csv
from main_modules.regexp import (validata_phone_number, validate_http_status, validate_inn, validate_identifier,
                                 validate_ipv4, validate_latitude, validate_blood_type, validate_isbn, validate_uuid,
                                 validate_date)

if __name__ == "__main__":
    settings = read_json("settings.json")
    serialize_result(14, "2006", settings["path_to_result"])
    data = read_csv(settings["path_to_data"])
    idx_of_bad_values = []
    print(data)
    for index, value in data["date"].items():
        if not validate_date(value):
            print(f"Index:{index}    Number:{value}")
            idx_of_bad_values.append(index)
    print(len(idx_of_bad_values))



