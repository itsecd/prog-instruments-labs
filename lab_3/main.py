from checksum import calculate_checksum, serialize_result
from file_manager import csv_open, open_json
from validation import validate_row


def main():
    config = open_json("settings.json")
    regex = open_json(config["regex_json"])
    data = csv_open(config["input_csv"])

    invalid_rows = [index for index, row in enumerate(data) if not validate_row(row, regex)]
    checksum = calculate_checksum(invalid_rows)
    serialize_result(config["variant"], checksum, config["result_json"])

    print("Statistics:")
    print(f"Total lines: {len(data)}")
    print(f"Invalid lines: {len(invalid_rows)}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()