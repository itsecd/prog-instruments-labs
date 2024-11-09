import re
import pandas as pd

from typing import List

import checksum


class CheckForm:
    REGEXES = {
        "telephone": r"[+]\d[-][(]\d{3}[)][-]\d{3}[-]\d{2}[-]\d{2}",
        "http_status_message": r"\d{3}\s\b[a-zA-Z\s]+",
        "inn": r"\d{12}",
        "identifier": r"\d{2}-\d{2}/\d{2}",
        "ip_v4": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        "latitude": r"-{0,1}\d{1,3}\.\d*",
        "blood_type": r"([AB]|[O]){1,2}[\+-]",
        "isbn": r"(\d{3}-){0,1}\d-\d{5}-\d{3}-\d",
        "uuid": r"[a-z0-9]{8}-([a-z0-9]{4}-){3}[a-z0-9]{12}",
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
    data = pd.read_csv("lab_3\\22.csv", sep=";", encoding="utf-16")
    error_rows = CheckForm.get_error_rows(data)
    check_sum = checksum.calculate_checksum(error_rows)
    checksum.serialize_result(22, check_sum)
