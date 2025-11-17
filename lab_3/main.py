import re
from checksum import calculate_checksum, serialize_result
from file_processing import read_json, read_csv
from typing import List


def validate_row(row: dict[str, str], patterns: dict[str, str]) -> bool:
    """
    Проверяет валидность строки данных на основе регулярных выражений.
    :param row: словарь с данными строки
    :param patterns: словарь с регулярными выражениями для валидации
    :return: True если все поля строки соответствуют регулярным выражениям, иначе False
    """
    for field, pattern in patterns.items():
        value = row.get(field)
        if not re.fullmatch(pattern, value):
            return False
    return True


def find_error_rows(rows: List[dict], patterns: dict[str, str]) -> List[int]:
    """
    Находит номера строк с ошибками валидации.
    :param rows: список словарей с данными строк для проверки
    :param patterns: словарь с регулярными выражениями для валидации полей
    :return: список номеров строк, в которых найдены ошибки валидации
    """
    return [i for i, row in enumerate(rows) if not validate_row(row, patterns)]


def main():
    source = read_json("settings.json")
    csv_file_path = source["CSV_FILE"]
    patterns = read_json(source["PATTERNS"])
    result_path = source["RESULT"]
    variant = int(csv_file_path[:-4])

    rows = read_csv(csv_file_path)
    error_rows = find_error_rows(rows, patterns)

    checksum = calculate_checksum(error_rows)
    serialize_result(variant, checksum, result_path)

    print(f"Всего строк: {len(rows)}")
    print(f"Строк с ошибками: {len(error_rows)}")
    print(f"Контрольная сумма: {checksum}")


if __name__ == "__main__":
    main()
