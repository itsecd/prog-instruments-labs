from filework import csv_open, json_open
from checksum import calculate_checksum, serialize_result
from validation import validate_row


def main():
    """
    Entry point to the program
    """
    config = json_open("config.json")

    regular_expressions = json_open(config["regular_expressions"])

    data = csv_open(config["input_csv"])

    invalid_rows = [
        index
        for index, row in enumerate(data)
        if not validate_row(row, regular_expressions)
    ]

    checksum = calculate_checksum(invalid_rows)

    serialize_result(config["var"], checksum, config["result_json"])

    print("Statistics:")
    print(f"Total lines: {len(data)}")
    print(f"Invalid lines: {len(invalid_rows)}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
