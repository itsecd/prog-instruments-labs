from main_modules.checksum import serialize_result
from main_modules.file_work import read_json, read_csv
from main_modules.regexp import validata_phone_number


if __name__ == "__main__":
    settings = read_json("settings.json")
    serialize_result(14, "2006", settings["path_to_result"])
    data = read_csv(settings["path_to_data"])





