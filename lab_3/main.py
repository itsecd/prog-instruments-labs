import json
from file import load_csv, load_json
from validate_serialaze import validate_row
from checksum import calculate_checksum, serialize_result


def main():
    settings = load_json("settings.json")

    regexps = load_json(settings["regular_file"])

    rows = load_csv(settings["input_file"])

    bad_rows = [idx for idx, row in enumerate(rows) if not validate_row(row, regexps)]

    checksum = calculate_checksum(bad_rows)

    serialize_result(settings["variant"], checksum, settings["result_file"])

    print(f"Всего строк: {len(rows)}")
    print(f"Ошибочных строк: {len(bad_rows)}")
    print(f"Контрольная сумма: {checksum}")

if __name__ == "__main__":
    main()
