import pandas as pd
from pandas import DataFrame


def read_csv(filename: str) -> DataFrame:
    df = pd.read_csv(filename, sep=';', encoding='utf-16', header=0)

    return df


def validity_check(data: DataFrame):
    pass

if __name__ == "__main__":
    data = read_csv("27.csv")
    print(data)
