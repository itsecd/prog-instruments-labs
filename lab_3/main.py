from read_write import read_json, read_csv
from validator import find_invalid_rows
from checksum import calculate_checksum, serialize_result


def main():

    config = read_json("config.json")

    data = read_csv(config.get('data'))
    patterns = read_json(config.get('patterns'))
    list_with_errors = find_invalid_rows(data, patterns)
    print(list_with_errors)
    checksum = calculate_checksum(list_with_errors)
    serialize_result(config.get('var'), checksum)

if __name__ == "__main__":
    main()