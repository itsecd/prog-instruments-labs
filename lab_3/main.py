import csv
import json
import re
from checksum import calculate_checksum, serialize_result
from pattern import (
    EMAIL, STATUS_MESSAGE, INN, PASSPORT, IPV4,
    LATITUDE, HEX_COLOR, ISBN, TIME, UUID
)


def read_csv(filename: str) -> list:
    encodings = ['utf-8', 'utf-16']

    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as file:
                reader = csv.reader(file, delimiter=';')
                data = list(reader)
                print(f" файл успешно прочитан: {encoding}")
                print(f"заголовки: {data[0] if data else 'нет данных'}")
                return data
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f" Ошибка: Файл {filename} не найден")
            return []

    print(f" Ошибка: Не удалось прочитать файл {filename}")
    return []


def create_validator(pattern: str, additional_check=None):
    def validator(value: str) -> bool:
        if not re.match(pattern, value):
            return False
        if additional_check:
            return additional_check(value)
        return True

    return validator


def latitude_range_check(lat: str) -> bool:
    try:
        return -90 <= float(lat) <= 90
    except ValueError:
        return False


validators = [
    create_validator(EMAIL),
    create_validator(STATUS_MESSAGE),
    create_validator(INN),
    create_validator(PASSPORT),
    create_validator(IPV4),
    create_validator(LATITUDE, latitude_range_check),
    create_validator(HEX_COLOR),
    create_validator(ISBN),
    create_validator(UUID),
    create_validator(TIME)
]


def validate_row(row: list) -> bool:
    return len(row) == 10 and all(
        validator(field) for validator, field in zip(validators, row)
    )



def find_invalid_rows(filename: str) -> list:
    data = read_csv(filename)
    if not data:
        return []

    return [
        i for i, row in enumerate(data[1:], start=0)
        if not validate_row(row)
    ]


def main():
    CSV_FILENAME = "21.csv"
    VARIANT_NUMBER = 21

    print("валидация данных")





    invalid_rows = find_invalid_rows(CSV_FILENAME)
    checksum = calculate_checksum(invalid_rows)

    print(f"\n - результаты -")
    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")

    serialize_result(VARIANT_NUMBER, checksum)
    print("Результат сохранен в result.json")

    if len(invalid_rows) == 1000:
        print(" Количество невалидных строк соответствует ")
    else:
        print(f" Найдено {len(invalid_rows)} строк, надо 1000")


if __name__ == "__main__":
    main()