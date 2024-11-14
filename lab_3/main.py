import json
import re
import os

import pandas as pd
import checksum as check

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