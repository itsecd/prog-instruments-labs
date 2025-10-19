from csv_processor import read_csv
from constants import CSV_DELIMITER,CSV_FILENAME

def main():
    print("Validation of the CSV file")
    data = read_csv(CSV_FILENAME,CSV_DELIMITER)
    print(data)

if __name__ == "__main__":
    main()