import re
import os

import pandas as pd
import checksum as check

PATTERNS = {
    "email" : r"^\w+(\.\w+)*@\w+(\.\w+)+$", ###
    "height" : r"^[1-2]\.\d{2}$", ###
    "snils" : r"^\d{11}$", ###
    "passport" : r"^(\d{2}\s){2}\d{6}$", ###
    "occupation" : r"^[а-яА-Яa-zA-ZёЁ\\s-]+$", ###
    "longitude" : r"^-?(180|1[0-7]\d|\d{1,2})(\.\d+)?$", ###
    "hex_color" : r"^#([a-f0-9]{6}|[a-f0-9]{3})$", ###
    "issn" : r"^\d{4}-\d{4}$", ###
    "locale_code" : r"^[a-z]{1,3}(-[a-z]+)?(-[a-z]{2})?$", ###
    "time" : r"^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\.(\d{1,6})$", ###
}


def check_row(row : pd.Series) -> bool:
    for name_column, value in zip(PATTERNS.keys(), row):
        pattern = PATTERNS[name_column]
        if not re.search(pattern, str(value)):
            return False
    return True


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
    list_index = get_invalid_index(
        os.path.join("lab_3", "59.csv")
    )
    print(len(list_index))
    result = check.calculate_checksum(list_index)
    check.serialize_result(59, result)