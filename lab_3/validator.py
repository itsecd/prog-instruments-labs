import pandas as pd
import re

def find_invalid_rows(df: pd.DataFrame, patterns: dict) -> list:
    """
    Находит номера строк с некорректными значениями
    :param df: DataFrame с данными
    :param patterns: словарь с паттернами для проверки
    :return: список номеров строк с ошибками
    """
    invalid_rows = set()

    for column, pattern in patterns.items():
        if column in df.columns:
            for idx, value in enumerate(df[column]):
                if pd.isna(value):
                    invalid_rows.add(idx)
                    continue

                if not re.match(pattern, str(value)):
                    invalid_rows.add(idx)

    return sorted(list(invalid_rows))