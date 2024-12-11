from workfiles import csv_to_list, get_invalid_list
from configurations import path, regex_patterns
from checksum import calculate_checksum


if __name__ == '__main__':
    check_list = []
    data = csv_to_list(path)
    check_list = get_invalid_list(regex_patterns, data)
    print(len(check_list))
    print(calculate_checksum(check_list))
