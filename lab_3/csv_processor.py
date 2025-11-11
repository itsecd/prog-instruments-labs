import csv
from typing import Any, Dict, List, Set
from validators import validate_cell

def read_csv(filename, delimiter) -> List[Dict[str, Any]]:
    try:
        with open(filename, 'r', encoding='utf-16') as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            data = list(reader)
            print(f"Read {len(data)} lines from the CSV file")
            return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def validate_data(data: List[Dict[str, Any]],
                  validation_patterns: Dict[str, Any]) -> Set[int]:
    """
    Validation of all CSV file data
    """
    invalid_rows = set()
    for row_index, row in enumerate(data, start=0):
        is_row_valid = True

        for column, pattern in validation_patterns.items():
            if column in row:
                value = row[column]
                if not validate_cell(value, pattern):
                    is_row_valid = False
                    break

        if not is_row_valid:
            invalid_rows.add(row_index)

    print(f"Found {len(invalid_rows)} invalid rows")
    return invalid_rows