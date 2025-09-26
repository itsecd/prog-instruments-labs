from main_modules.checksum import serialize_result
from main_modules.file_work import read_json, read_csv
from main_modules.regexp import validata_phone_number, validate_http_status



if __name__ == "__main__":
    settings = read_json("settings.json")
    serialize_result(14, "2006", settings["path_to_result"])
    data = read_csv(settings["path_to_data"])
    bad_http_statuses = []
    print(data)
    for index, phone_number in data["http_status_message"].items():
        if not validate_http_status(phone_number):
            print(f"Index:{index}    Number:{phone_number}")
            bad_http_statuses.append(index)
print(len(bad_http_statuses))


