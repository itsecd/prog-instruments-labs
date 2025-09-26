from main_modules.file_work import read_json
from main_modules.checksum import serialize_result


if __name__ == "__main__":
    settings = read_json("settings.json")
    serialize_result(14, "2006", settings["path_to_result"])

