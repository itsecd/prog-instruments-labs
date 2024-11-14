import json
import re
import os

import pandas as pd
import checksum as check

PATTERNS = {
    "email" : r"^\w+(\.\w+)*@\w+(\.\w+)+$",
    "height" : r"^[1-2]\.\d{2}$",
    "snils" : r"^\d{11}$",
    "passport" : r"^\d{2}\s\d{2}\s\d{6}$",
    "occupation" : r"^[а-яА-Яa-zA-Z\s-]*$",
    "longitude" : r"^-?(180|1[0-7]\d|\d{1,2})(\.\d+)?$",
    "hex_color" : r"#\d{6}",
    "issn" : r"\d{4}-\d{4}",
    "locale_code" : r"^[a-z]{1,3}(-[a-z]+)?(-[a-z]{2})?$",
    "time" : r"^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.(\d{1,6})$"
}

def check_row():
    pass

def open_csv(path : str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-16", sep=";")
    return data

def get_invalid_index(path : str) -> list:
    data = open_csv(path)
    invalid_index_list = list()
    for index, row in data.iterrows():
        if not check_row(row):
            invalid_index_list.append(index)
    return invalid_index_list

if __name__ == "__main__":
    pass