import csv
import json

def open_json(path):
    try:
        with open (path, "r", encoding = 'utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found in this path: '{path}'")
    except json.JSONDecodeError as e:
        raise(f"JSON decoding error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error")
    
def save_json(data, path):
    try:
        with open (path, 'w', encoding = 'utf-8') as file:
            json.dump(data, file, ensure_ascii = False, indent = 4)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found in this path: '{path}'")
    except Exception as e:
        raise Exception(f"Unexpected error")
    

def open_csv(path):
    try:
        with open(path, 'r', encoding = 'utf-16', newline = "") as f:
            text = csv.DictReader(f, delimiter=";")
            return list(text)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found from path '{path}'")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            "utf-16",
            b"",
            0,
            1,
            f"Failed to decode file '{path}' with encoding utf-16: {e}",
        )
    except csv.Error as e:
        raise csv.Error(f"CSV error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error in '{path}': '{e}'")


