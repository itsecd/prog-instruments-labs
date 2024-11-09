import re
import pandas as pd

from typing import List

import checksum


class CheckForm:
    REGEXES = {
        "telephone": r"[+]7[-]\(\d{3}\)-\d{3}-\d{2}-\d{2}",
        "http_status_message": r"\d{3}\s\b[a-zA-Z\s]+",
        "inn": r"\d{10}|\d{12}",
        "identifier": r"\d{2}-\d{2}/\d{2}",
        "ip_v4": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        "latitude": r"-{0,1}\d{1,2}\.\d*",
        "blood_type": r"(A|B|AB|O)[+−]{1}",
        "isbn": r"(\d{3}-){0,1}\d-\d{5}-\d{3}-\d",
        "uuid": r"[a-f\d]{8}-([a-f\d]{4}-){3}[a-f\d]{12}",
        "date": r"\d{4}-\d{2}-\d{2}",
    }

    @classmethod
    def get_error_rows(cls, data: pd.DataFrame) -> List[int]:
        """Проходится по строкам датафрейма и проверяет записи
        на совпадение с форматом с помощью регулярных выражений.

        Args:
            data (pd.DataFrame): датафрейм со строками для анализа

        Returns:
            List[int]: список номеров строк, сожержащих строку с некорректным форматом
        """
        fail_rows = []
        keys = list(data.keys())
        for i, row in data.iterrows():
            for j in range(len(row)):
                is_correct_form = re.fullmatch(cls.REGEXES[keys[j]], row.iloc[j])
                if not is_correct_form:
                    fail_rows.append(i)
                    break
        return fail_rows
        


if __name__ == "__main__":
    FILE_PATH = "lab_3\\22.csv"
    data = pd.read_csv(FILE_PATH, sep=";", encoding="utf-16")
    error_rows = CheckForm.get_error_rows(data)
    check_sum = checksum.calculate_checksum(error_rows)
    if check_sum != "048c9d1978185a45af355149a26625f5":
        print(check_sum)
        print("Not okay!")
    else:
        print("Okay!")
    checksum.serialize_result(22, check_sum)
