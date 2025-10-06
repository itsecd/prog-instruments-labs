import pandas as pd

from pathlib import Path

def get_csv_path():
    csv_path = Path(__file__).parent.parent/ 'data' / '6.csv'
    return csv_path

def load_csv(path: str):
    try:
        with open(path, mode="r", encoding='utf-16') as file:
            return pd.read_csv(file, delimiter=";")

    except FileNotFoundError as file_not_found:
        raise FileNotFoundError(f"File was not found: {file_not_found}")

    except Exception as e:
        raise Exception(f"Error reading csv file: {e}")
