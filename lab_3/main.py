from checksum import calculate_checksum, serialize_result
from patterns import PATTERNS
from utils import read_csv_file, find_invalid_rows_in_table


def main():
    matrix = read_csv_file("14.csv")
    invalid_rows = find_invalid_rows_in_table(matrix, PATTERNS)
    checksum = calculate_checksum(list(invalid_rows))
    serialize_result(14, checksum)


if __name__ == "__main__":
    main()