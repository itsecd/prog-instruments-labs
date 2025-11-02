import re
import csv
import json
from checksum import calculate_checksum, serialize_result


with open('patterns.json', 'r', encoding='utf-8') as f:
    PATTERNS = json.load(f)


def validate_field(field_name, value):
    """Function for validating a field by its name"""

    # Regular expression check
    if not re.match(PATTERNS[field_name], value):
        return False

    # Additional checks for specific fields
    if field_name == "http_status":
        code = value.split()[0]
        return code.isdigit() and 100 <= int(code) <= 599

    elif field_name == "ip_v4":
        blocks = value.split('.')
        for block in blocks:
            if not block.isdigit() or not (0 <= int(block) <= 255):
                return False
        return True

    elif field_name == "latitude":
        try:
            lat_float = float(value)
            return -90.0 <= lat_float <= 90.0
        except (ValueError, TypeError):
            return False

    elif field_name == "isbn":
        clean_isbn = value.replace(' ', '')
        digits = re.sub(r'[^\d]', '', clean_isbn)
        return len(digits) in [10, 13]

    elif field_name == "time":
        parts = value.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_part = parts[2]

        if '.' in seconds_part:
            seconds = int(seconds_part.split('.')[0])
        else:
            seconds = int(seconds_part)

        return (0 <= hours <= 23) and (0 <= minutes <= 59) and (0 <= seconds <= 59)

    return True


def validate_row(row):
    """Checking the entire string for validity"""
    field_names = [
        "email",
        "http_status",
        "inn",
        "passport",
        "ip_v4",
        "latitude",
        "hex_color",
        "isbn",
        "uuid",
        "time"
    ]

    for i, field_name in enumerate(field_names):
        if not validate_field(field_name, row[i]):
            return False
    return True


def process_csv_file(filename):
    """Processing a CSV file and finding lines with errors"""
    error_rows = []

    with open(filename, 'r', encoding='utf-16') as file:
        # Skip the title
        reader = csv.reader(file, delimiter=';')
        next(reader)

        for row_number, row in enumerate(reader):
            if not validate_row(row):
                error_rows.append(row_number)

    return error_rows


def main():
    # Processing the file
    error_rows = process_csv_file('13.csv')

    # Calculating the checksum
    checksum = calculate_checksum(error_rows)

    # Display the results
    print(f"Lines with errors found: {len(error_rows)}")
    print(f"Checksum: {checksum}")

    # Serialize the result
    serialize_result(13, checksum)

    # Save in result.json
    result_data = {
        "variant": 13,
        "checksum": checksum
    }

    with open('result.json', 'w', encoding='utf-8') as file:
        json.dump(result_data, file, indent=2)


if __name__ == "__main__":
    main()
