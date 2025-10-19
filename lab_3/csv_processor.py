import csv
from typing import List, Dict, Any

def read_csv(filename,delimiter) -> List[Dict[str, Any]]:
    """
    Reading a CSV file and returning data
    :param delimiter:
    :param filename:
    :return:
    """
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
