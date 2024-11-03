import re
import pandas as pd
import checksum as cs


FILE_PATH = r"C:\Users\minh\PycharmProjects\prog-instruments-labs\lab_3\84.csv"

PATTERN = {
    "telephone" : "^\\+7-\\(\\d{3}\\)-\\d{3}-\\d{2}-\\d{2}$",
    "height" : "^[0-2]\\.\\d{2}$",
    "snils" : "^\\d{11}$",
    "identifier" : "^\\d{2}\\-\\d{2}/\\d{2}$",
    "occupation" : "^[a-zA-Zа-яА-ЯёЁ\\s-]+$",
    "longitude" : "^-?(180(\\.0+)?|1[0-7]\\d(\\.\\d+)?|[1-9]?\\d(\\.\\d+)?)$",
    "blood_type" : "^(A|B|AB|O)[+−]$",
    "issn" : "^\\d{4}-\\d{4}$",
    "locale_code" : "^[a-z]+(-[a-z]+)*$",
    "date" : "^\\d{4}-\\d{2}-\\d{2}$"
}


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Чтение данных из CSV-файла и возврат DataFrame.
    :param file_path: Путь к CSV-файлу.
    :return: DataFrame, содержащий данные из CSV-файла, или None в случае ошибки.
    """
    try:
        data = pd.read_csv(file_path, encoding="utf-16", sep=";")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    return data


def check_row(row: pd.Series) -> bool:
    """
    Проверка корректности строки данных на основе регулярных выражений.
    :param row: Строка данных в формате pd.Series.
    :return: True, если строка корректна, иначе False.
    """
    return all(re.match(PATTERN[key], str(row[key])) for key in PATTERN if key in row)


def make_data_invalid(file_path: str) -> list:
    """
    Определение индексов некорректных строк в CSV-файле.
    :param file_path: Путь к CSV-файлу.
    :return: Список индексов некорректных строк.
    """
    data = read_csv(file_path)
    if data is None:
        return []
    invalid = []
    for index, row in data.iterrows():
        if not check_row(row):
            invalid.append(index)
    return invalid


if __name__ == "__main__":
    cs.serialize_result(84, cs.calculate_checksum(make_data_invalid(FILE_PATH)))