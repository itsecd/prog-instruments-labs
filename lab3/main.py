import argparse

from filehandler.filehandler import read_data, read_csv
from regular_expression.regular_expression import check_validate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", type=str, help="directory to settings file")
    return parser.parse_args()


def main():
    try:
        settings = read_data(get_args().settings_file)
        regexp = read_data(settings["regexp"])
        # table = read_csv(settings["csv_file"])
        # print(table)
        print(f"Phone number: {check_validate("+7-(969)-765-17-05", regexp["telephone"])}")
        print(f"HTTP 1st var: {check_validate("200 OK", regexp["http_status_message"])}")
        print(f"HTTP 2nd var: {check_validate("226 IM Used", regexp["http_status_message"])}")
        print(f"INN 12 symbols: {check_validate("733499833600", regexp["inn"])}")
        print(f"INN more than 12 symbols: {check_validate("7334998336009999", regexp["inn"])}")
        print(f"Identifier: {check_validate("62-71/26", regexp["identifier"])}")
        print(f"Identifier: {check_validate("62 71/26", regexp["identifier"])}")
        print(f"IP_v4: {check_validate("19.121.223.58", regexp["ip_v4"])}")
        print(f"IP_v4: {check_validate("19.121223.58", regexp["ip_v4"])}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()