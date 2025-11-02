import re
import csv
import json
from checksum import calculate_checksum, serialize_result


def validate_email(email):
    """Email validity check"""
    pattern = r'^[a-zA-Z0-9._]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_http_status(status):
    """HTTP status validity check"""
    pattern = r'^\d{3} [A-Za-z ]+$'
    if not re.match(pattern, status):
        return False

    # Additional status code check
    code = status.split()[0]
    return code.isdigit() and 100 <= int(code) <= 599


def validate_inn(inn):
    """INN  validity check(12 numbers)"""
    pattern = r'^\d{12}$'
    return bool(re.match(pattern, inn))


def validate_passport(passport):
    """Passport  validity check(format: XX XX XXXXXX)"""
    pattern = r'^\d{2} \d{2} \d{6}$'
    return bool(re.match(pattern, passport))


def validate_ip_v4(ip):
    """IPv4 adress validity check"""
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(pattern, ip):
        return False

    # Check that each block is in the range 0-255
    blocks = ip.split('.')
    for block in blocks:
        if not (0 <= int(block) <= 255):
            return False
    return True


def validate_latitude(lat):
    """Latitude  validity check(-90 to 90)"""
    try:
        lat_float = float(lat)
        return -90.0 <= lat_float <= 90.0
    except (ValueError, TypeError):
        return False


def validate_hex_color(color):
    """HEX color validity check"""
    pattern = r'^#[0-9a-fA-F]{6}$'
    return bool(re.match(pattern, color))


def validate_isbn(isbn):
    """ISBN  validity check"""
    pattern = r'^\d{1,5}-\d-\d{5}-\d{3}-\d$|' \
              r'^\d{1,5}-\d{5}-\d{3}-\d$|' \
              r'^\d{3}-\d-\d{3}-\d{5}-\d$'
    if re.match(pattern, isbn):
        # Check the count of numbers
        digits = re.sub(r'[^\d]', '', isbn)
        return len(digits) in [10, 13]


def validate_uuid(uuid_str):
    """UUID validity check"""
    pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    return bool(re.match(pattern, uuid_str))


def validate_time(time_str):
    """Time validity check"""
    pattern = r'^([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]{1,6})?$'
    if not re.match(pattern, time_str):
        return False

    # Additional check of time components
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_part = parts[2]

    if '.' in seconds_part:
        seconds = int(seconds_part.split('.')[0])
    else:
        seconds = int(seconds_part)

    return (0 <= hours <= 23) and (0 <= minutes <= 59) and (0 <= seconds <= 59)


def validate_row(row):
    """Checking the entire string for validity"""
    validators = [
        validate_email,
        validate_http_status,
        validate_inn,
        validate_passport,
        validate_ip_v4,
        validate_latitude,
        validate_hex_color,
        validate_isbn,
        validate_uuid,
        validate_time
    ]

    for i, validator in enumerate(validators):
        if not validator(row[i]):
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

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
