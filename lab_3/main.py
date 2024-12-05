import re
import pandas as pd
import json


from regulars import CSV_FILE_PATH, JSON_PATH, REGULAR

from checksum import calculate_checksum


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file.

    :param file_path: The path to the CSV file as a string.
    :return: A pandas DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path, encoding="utf-16", sep=";", header=0)
    return data


def write_to_file(variant: int, checksum: str) -> None:
    """
    Serializes the results of the lab work into the result.json file.

    :param variant: The variant number of the lab work.
    :param checksum: The checksum calculated using calculate_checksum().
    """
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    data["variant"] = variant
    data["checksum"] = checksum
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)


def validate_data(file_path: str) -> list:
    """
    Loads, validates the data, and calculates the checksum.
    :param file_path: The path to the CSV file as a string.
    :return: A list of indices of rows that contain errors.
    """
    data = load_csv(file_path)
    data.columns = [f"regular{i+1}" for i in range(10)]
    error = []
    for index, row in data.iterrows():
        row_has_error = False
        for col, pattern in REGULAR.items():
            if not re.match(pattern, str(row[col])):
                row_has_error = True
                break
        if row_has_error:
            error.append(index)

    return error


if __name__ == "__main__":
    result = validate_data(CSV_FILE_PATH)
    checksum = calculate_checksum(result)
    write_to_file(variant="8", checksum=checksum)
    print("Контрольная сумма:", checksum)
    print(f"Невалидные записи: {len(result)}")