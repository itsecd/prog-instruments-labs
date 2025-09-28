from main_modules.checksum import serialize_result, calculate_checksum
from main_modules.file_work import read_json, read_csv
from main_modules.regexp import get_rows_with_mistakes

if __name__ == "__main__":
    settings = read_json("settings.json")
    patterns_dict = read_json(settings["path_to_patterns"])
    patterns_list = list(patterns_dict.values())
    data = read_csv(settings["path_to_data"])
    rows_with_mistakes = get_rows_with_mistakes(data, patterns_list)
    checksum = calculate_checksum(rows_with_mistakes)
    serialize_result(settings["variant"], checksum, settings["path_to_result"])



