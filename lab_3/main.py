from data_check import check_data
from checksum import calculate_checksum, serialize_result
from cons import FILE_PATH_PATTERNS, FILE_PATH_CSV, VAR


if __name__ == "__main__":
    invalid = check_data(FILE_PATH_CSV, FILE_PATH_PATTERNS)
    check_sum = calculate_checksum(invalid)
    serialize_result(VAR, check_sum)
 