import re
import os

import pandas as pd
import checksum as check

CSV_PATH = "53.csv"
REGULARS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "http_status_message": r"^\d{3} [A-Za-z ]+$",
    "inn": r"",
    "passport": r"^\d{2}\s\d{2}\s\d{6}$",
    "ip_v4": r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|"
             r"[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$",
    "latitude": r"^\-?(180|1[0-7][0-9]|\d{1,2})\.\d+$",
    "hex_color": r"^#[0-9a-fA-F]{6}$",
    "isbn": r"(\d{3}-)?\d-(\d{5})-(\d{3})-\d",
    "uuid": r"",
    "time": r"^\d{2}:\d{2}:\d{2}\.\d{6}$"
}


def open_csv(path: str) -> pd.DataFrame:
    """
    function open csv file and get out info
    :param path: str
    :return: DataFrame
    """
    try:
        data = pd.read_csv(path, encoding="utf-16", sep=";")
        return data
    except Exception as e:
        print(f"Ошибка при открытия файла {CSV_PATH}: {e}")
        raise


def get_invalid_index(path: str) -> list:
    """
    function get list of index invalid rows
    :param path:
    :return: list
    """
    data = open_csv(path)
    index_list = list()
    for index, row in data.iterrows():
        if not check_invalid_row(row):
            index_list.append(index)
    return index_list


def check_invalid_row(row: pd.Series) -> bool:
    """
    Check row and return invalid it or no
    :param row: pd.Series
    :return: bool
    """
    for name_column, value in zip(REGULARS.keys(), row):
        pattern = REGULARS[name_column]
        if not re.search(pattern, str(value)):
            return False
    return True


if __name__ == "__main__":
    list_index = get_invalid_index(CSV_PATH)
    print(len(list_index))
    result = check.calculate_checksum(list_index)
    check.serialize_result(53, result)