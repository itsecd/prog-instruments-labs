import argparse

from checksum import calculate_checksum, serialize_result
from filehandler.filehandler import read_csv, read_data
from regular_expression.regular_expression import find_invalid_rows


def get_args():
    """
    The function parses directory to settings json file from terminal
    :return: directory to settings json file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", type=str, help="directory to settings file")
    return parser.parse_args()


def main():
    try:
        settings = read_data(get_args().settings_file)
        regexp = read_data(settings["regexp"])
        table = read_csv(settings["csv_file"])
        indexes = find_invalid_rows(table, regexp)
        checksum = calculate_checksum(list(indexes))
        serialize_result(settings["variant"], checksum, settings["result"])
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
