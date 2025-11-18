from checksum import calculate_checksum, serialize_result
from constants import CSV_DELIMITER, CSV_FILENAME, VARIANT
from csv_processor import read_csv, validate_data
from validators import get_validation_patterns



def main():
    print("Validation of the CSV file")

    data = read_csv(CSV_FILENAME, CSV_DELIMITER)
    if not data:
        print("No data loaded. Exiting.")
        return

    validation_patterns = get_validation_patterns()
    print(f"Loaded {len(validation_patterns)} validation patterns")

    invalid_rows = validate_data(data, validation_patterns)
    print(f"Invalid rows: {invalid_rows}")

    invalid_row_numbers = list(invalid_rows)
    checksum = calculate_checksum(invalid_row_numbers)
    print(f"Calculated checksum: {checksum}")

    variant = VARIANT
    serialize_result(variant, checksum)
    print(f"Result saved to result.json for variant {variant}")


if __name__ == "__main__":
    main()