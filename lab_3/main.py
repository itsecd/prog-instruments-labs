from csv_processor import read_csv, validate_data
from validators import get_validation_patterns
from constants import CSV_DELIMITER,CSV_FILENAME

def main():
    print("Validation of the CSV file")
    data = read_csv(CSV_FILENAME,CSV_DELIMITER)
    validation_patterns = get_validation_patterns()
    print(f"Loaded {len(validation_patterns)} validation patterns")
    invalid_rows = validate_data(data, validation_patterns)
    print(invalid_rows)

if __name__ == "__main__":
    main()