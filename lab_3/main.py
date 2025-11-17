from checksum import calculate_checksum
from filework import read_csv, read_json, write_json
from validate import invalid_validation_rows


def main() -> None:
    patterns = read_json('patterns.json')
    rows = read_csv('74.csv')
    invalid_rows = invalid_validation_rows(rows, patterns)
    checksum = calculate_checksum(invalid_rows)

    result = {"variant": 74, "checksum": checksum}
    write_json(result, 'result.json')


if __name__ == "__main__":
    main()
