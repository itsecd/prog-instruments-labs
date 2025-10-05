from lab_3.checksum import calculate_checksum, serialize_result
from lab_3.dataprocessing import read_json, read_csv, validity_check


def main():
    try:
        settings = read_json("settings.json")
        data = read_csv(settings["csv_filename"])
        checksum = calculate_checksum(validity_check(data, settings["regular_expressions"]))
        serialize_result(settings["variant"], checksum, settings["result_filename"])
    except Exception as exc:
        print(f'Something went wrong: {exc}')


if __name__ == "__main__":
    main()
